TASKS THIS WEEK
- read the paper
- get the idea

Team 2 working on the equation:
- Navier Stokes Alpha Equation

---
### Background
The traditional Navier-Stokes (NS) equations are used to model all manner of fluid movement from blood flow in veins to water flow in rivers to air flow in the atmosphere. As such, they can be considered a more elaborate retelling of two of Newton's laws: the conservation of mass and momentum. They are presented here for reference:
$$
    \left\{
    \begin{alignedat}{3} 
        &\frac{\partial u}{\partial t}+(u\cdot \nabla)u=-\nabla p + \nu \Delta u + f,\\
        & \nabla \cdot u = 0
    \end{alignedat}
    \right.
$$
where:
$u$: velocity vector field
$t$: time
$\nabla$: gradient operator ($\nabla f = (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z})$)
$p$: pressure field
$\nu$: viscosity
$\Delta$: Laplace operator ($\Delta f= (\frac{\partial^{2} f}{\partial x^{2}}, \frac{\partial^{2} f}{\partial y^{2}}, \frac{\partial^{2} f}{\partial z^{2}})$)
$f$: outside forces (like gravity, etc)


## NS-Alpha
The Navier-Stokes-$\alpha$ (NS-$\alpha$) equations are defined by Albanez & Benvenutti as the following:
$$
    \left\{
    \begin{alignedat}{3} 
        &\frac{\partial v}{\partial t}-u \times (\nabla \times v) - \nu \Delta v + \nabla p = f, \\
        &v = (1-\alpha^{2} \Delta)u\\
        & \nabla \cdot u = 0
    \end{alignedat}
    \right.
$$
This is a modification of the traditional Navier-Stokes (NS) equations through the introduction of $v$ as a filtered alternative of the velocity field $u$. The second equation of this system highlights that $v$ is defined as being directly defined as a "percentage" of $u$, the size of which is determined by our length scale parameter $\alpha$. This filter is called a Helmholtz filter, and allows calculations to be much simpler for any computer working with it. Effectively, it reduces the spatial resolution of the problem domain, so that computational resources aren't being wasted on any  eddies or vortices smaller than the defined $\alpha$ value.