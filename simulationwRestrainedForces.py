from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from parmed import unit as u
from sys import stdout
from timeit import default_timer as timer
from omm_vfswitch import *
from copy import *
def heavyAtomsRest(system, xyz, atoms, resForce) :
    res_force = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    res_forceMagnitude = resForce * kilocalories_per_mole/nanometers**2
    res_force.addGlobalParameter('k', res_forceMagnitude)
    res_force.addPerParticleParameter("x0")
    res_force.addPerParticleParameter("y0")
    res_force.addPerParticleParameter("z0")
    for index, (atom_xyz, atom) in enumerate(zip(xyz, atoms)) :
        if atom.name in ('N', 'C', 'O', 'CA') :
            res_force.addParticle(index, atom_xyz.value_in_unit(nanometers))
        post = deepcopy(system)
        post.addForce(res_force)
        return post

print('Parsing the PSF file...')
psf = CharmmPsfFile('input.psf')

print('Parsing the PDB file...')
pdb = PDBFile('input.pdb')

print('Generating the topology...')
topol = pdb.topology

print('Setting PBC...')
periodic_box = topol.getPeriodicBoxVectors()
psf.setBox(periodic_box.value_in_unit(u.nanometers)[0][0], periodic_box.value_in_unit(u.nanometers)[1][1], periodic_box.value_in_unit(u.nanometers)[2][2])
print(psf.boxVectors)

#Parameter files
print('Reading the parameter files...')
params = CharmmParameterSet('toppar/par_all36m_prot.prm', 'toppar/toppar_water_ions-tamay.str', 'toppar/par_all36_lipid.prm', 'toppar/par_all36_na.prm', 'toppar/par_all36_cgenff.prm', 'toppar/toppar_all36_prot_na_combined.str', 'toppar/toppar_all36_lipid_prot.str', 'toppar/top_all36_prot.rtf', 'toppar/top_all36_lipid.rtf', 'toppar/top_all36_na.rtf', 'toppar/top_all36_cgenff.rtf', 'toppar/toppar_all36_carb_glycopeptide.str')

#Non-bonded interactions
system = psf.createSystem(params, nonbondedMethod=PME, nonbondedCutoff=1.2*u.nanometers, constraints=HBonds, ewaldErrorTolerance=0.00001 )
system = vfswitch(system, psf, 1.0, 1.2)

#Integrator
post = heavyAtomsRest(system, pdb.positions, pdb.topology.atoms(), 500)
integrator = LangevinIntegrator(310.15*u.kelvin, 1/u.picoseconds, 0.002*u.picoseconds)

#Platform
platform = Platform.getPlatformByName('CUDA')
prop = dict(DeviceIndex='0', Precision='single')
simulation = Simulation(psf.topology, post, integrator, platform, prop)
simulation.context.setPositions(pdb.positions)

#Minimization
print('Minimization...')
simulation.minimizeEnergy(maxIterations=5000)
#Equilibration
print('Equilibration...')
simulation.context.setVelocitiesToTemperature(310.15*u.kelvin)
print('Equilibration in NVT...')
simulation.reporters.append(
        StateDataReporter('eqNVT.csv', 1000, step=True,
        time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        separator='\t'))
simulation.step(50000)
print('Equilibration in NPT...')
system.addForce(MonteCarloBarostat(1.01325*u.bar, 310.15*u.kelvin, 25))
simulation.context.reinitialize(True)
simulation.reporters.append(
        StateDataReporter('eqNPT.csv', 1000, step=True,
        time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        separator='\t'))
simulation.reporters.append(app.DCDReporter('eqNPT.dcd', 10000))
simulation.reporters.append(CheckpointReporter('eqNPT.chk', 10000))
simulation.saveState('eqNPT.xml')
simulation.saveState(('eqNPT.state'))
simulation.saveCheckpoint(('eqNPT.chk'))
simulation.step(50000)
print('Loading equilibrated states...')
simulation.loadCheckpoint(('eqNPT.chk'))
eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
positions = eq_state.getPositions()
velocities = eq_state.getVelocities()
integrator = LangevinIntegrator(310.15*u.kelvin, 1/u.picoseconds, 0.002*u.picoseconds)
print('No restrained forces...')
post = heavyAtomsRest(system, pdb.positions, pdb.topology.atoms(), 0)
simulation = Simulation(psf.topology, post, integrator, platform, prop)
simulation.context.setPositions(positions)
simulation.context.setVelocities(velocities)
print('Running...')
simulation.reporters.append(
        StateDataReporter('production.csv', 10000, step=True,
        time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        separator='\t'))
simulation.reporters.append(app.DCDReporter('production.dcd', 10000))
simulation.reporters.append(CheckpointReporter('production.chk', 10000))
simulation.saveState('production.xml')
simulation.saveState(('production.state'))
simulation.saveCheckpoint(('production.chk'))
start=timer()
simulation.step(500000000)
print('Finished')
end=timer()
print(end-start)
