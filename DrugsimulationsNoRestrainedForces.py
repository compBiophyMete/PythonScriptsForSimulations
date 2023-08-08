from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from parmed import unit as u
from sys import stdout
from timeit import default_timer as timer
from omm_vfswitch import *
import numpy as np
import os
print('Loading the parameter files...')
toppar_filename = 'toppar.str'
workDir='/pwd/'
toppar = os.path.join(workDir, str(toppar_filename))
toppar_check = os.path.exists(toppar)
if  toppar_check == True:
    print("####Files loaded succesfully!#####")
else:
    print("###ERROR!####")

def read_toppar(filename,path):
        extlist = ['rtf', 'prm', 'str']

        parFiles = ()
        for line in open(filename, 'r'):
                if '!' in line: line = line.split('!')[0]
                parfile = line.strip()
                if len(parfile) != 0:
                        ext = parfile.lower().split('.')[-1]
                        if not ext in extlist: continue
                        parFiles += ( path + parfile, )

        params = CharmmParameterSet( *parFiles )
        return params, parFiles
print('Parsing the PSF file...')
psf = CharmmPsfFile('input.psf')
print('Parsing the PDB file...')
pdb = PDBFile('input.pdb')
print('Generating the topology...')
topol = pdb.topology
print('Setting PBC...')
xyz = np.array(pdb.positions/u.nanometers)
xyz[:,0] -= np.amin(xyz[:,0])
xyz[:,1] -= np.amin(xyz[:,1])
xyz[:,2] -= np.amin(xyz[:,2])
pdb.positions = xyz*u.nanometers
psf.setBox(8.4*u.nanometers, 8.4*u.nanometers, 8.4*u.nanometers)
#Parameter files
print('Reading the parameter files...')
params, directory = read_toppar(toppar, workDir)
#Non-bonded interactions
system = psf.createSystem(params, nonbondedMethod=PME, nonbondedCutoff=1.2*u.nanometers, constraints=HBonds, ewaldErrorTolerance=0.00001 )
system = vfswitch(system, psf, 1.0, 1.2)
#Thermostat and Barostat
system.addForce(MonteCarloBarostat(1.01325*u.bar, 310.15*u.kelvin, 25))
#Integrator
integrator = LangevinIntegrator(310.15*u.kelvin, 1/u.picoseconds, 0.002*u.picoseconds)
#Platform
platform = Platform.getPlatformByName('CUDA')
prop = dict(DeviceIndex='0', Precision='single')
simulation = Simulation(psf.topology, system, integrator, platform, prop)
simulation.context.setPositions(pdb.positions)
#Minimization
print('Minimization...')
simulation.minimizeEnergy(maxIterations=1000)
#Equilibration
print('Equilibration...')
simulation.context.setVelocitiesToTemperature(310.15*u.kelvin)
simulation.step(1000)
#Simulation
simulation.reporters.append(
        StateDataReporter('production.csv', 10000, step=True,
        time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        separator='\t'))
simulation.reporters.append(app.DCDReporter('production.dcd', 10000))
simulation.reporters.append(CheckpointReporter('production.chk', 10000))
simulation.saveState('production.xml')
print('Running...')
start=timer()
simulation.step(10000000)
print('Finished')
end=timer()
print(end-start)
