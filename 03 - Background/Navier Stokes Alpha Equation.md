### Intro
The **Navier–Stokes equations** ([/nævˈjeɪ stoʊks/](https://en.wikipedia.org/wiki/Help:IPA/English "Help:IPA/English") [_nav-YAY STOHKS_](https://en.wikipedia.org/wiki/Help:Pronunciation_respelling_key "Help:Pronunciation respelling key")) are [partial differential equations](https://en.wikipedia.org/wiki/Partial_differential_equation "Partial differential equation") which describe the motion of [viscous fluid](https://en.wikipedia.org/wiki/Viscous_fluid "Viscous fluid") substances. They were named after French engineer and physicist [Claude-Louis Navier](https://en.wikipedia.org/wiki/Claude-Louis_Navier "Claude-Louis Navier") and the Irish physicist and mathematician [George Gabriel Stokes](https://en.wikipedia.org/wiki/Sir_George_Stokes,_1st_Baronet "Sir George Stokes, 1st Baronet"). They were developed over several decades of progressively building the theories, from 1822 (Navier) to 1842–1850 (Stokes).

The Navier–Stokes equations mathematically express [momentum](https://en.wikipedia.org/wiki/Momentum "Momentum") balance for [Newtonian fluids](https://en.wikipedia.org/wiki/Newtonian_fluid "Newtonian fluid") and make use of [conservation of mass](https://en.wikipedia.org/wiki/Conservation_of_mass "Conservation of mass"). They are sometimes accompanied by an [equation of state](https://en.wikipedia.org/wiki/Equation_of_state "Equation of state") relating [pressure](https://en.wikipedia.org/wiki/Pressure "Pressure"), [temperature](https://en.wikipedia.org/wiki/Temperature "Temperature") and [density](https://en.wikipedia.org/wiki/Density "Density").[[1]](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#cite_note-1) They arise from applying [Isaac Newton's second law](https://en.wikipedia.org/wiki/Newton%27s_second_law "Newton's second law") to [fluid motion](https://en.wikipedia.org/wiki/Fluid_dynamics "Fluid dynamics"), together with the assumption that the [stress](https://en.wikipedia.org/wiki/Stress_\(mechanics\) "Stress (mechanics)") in the fluid is the sum of a [diffusing](https://en.wikipedia.org/wiki/Diffusion "Diffusion") [viscous](https://en.wikipedia.org/wiki/Viscosity "Viscosity") term (proportional to the [gradient](https://en.wikipedia.org/wiki/Gradient "Gradient") of velocity) and a [pressure](https://en.wikipedia.org/wiki/Pressure "Pressure") term—hence describing _viscous flow_. The difference between them and the closely related [Euler equations](https://en.wikipedia.org/wiki/Euler_equations_\(fluid_dynamics\) "Euler equations (fluid dynamics)") is that Navier–Stokes equations take [viscosity](https://en.wikipedia.org/wiki/Viscosity "Viscosity") into account while the Euler equations model only [inviscid flow](https://en.wikipedia.org/wiki/Inviscid_flow "Inviscid flow"). As a result, the Navier–Stokes are an [elliptic equation](https://en.wikipedia.org/wiki/Elliptic_partial_differential_equation "Elliptic partial differential equation") and therefore have better analytic properties, at the expense of having less mathematical structure (e.g. they are never [completely integrable](https://en.wikipedia.org/wiki/Completely_integrable "Completely integrable")).

The Navier–Stokes equations are useful because they describe the physics of many phenomena of [scientific](https://en.wikipedia.org/wiki/Scientific "Scientific") and [engineering](https://en.wikipedia.org/wiki/Engineering "Engineering") interest. They may be used to [model](https://en.wikipedia.org/wiki/Model_\(abstract\) "Model (abstract)") the weather, [ocean currents](https://en.wikipedia.org/wiki/Ocean_current "Ocean current"), water [flow in a pipe](https://en.wikipedia.org/wiki/Flow_conditioning "Flow conditioning") and air flow around a [wing](https://en.wikipedia.org/wiki/Airfoil "Airfoil"). The Navier–Stokes equations, in their full and simplified forms, help with the design of [aircraft](https://en.wikipedia.org/wiki/Aircraft_design_process#Preliminary_design_phase "Aircraft design process") and cars, the study of [blood flow](https://en.wikipedia.org/wiki/Hemodynamics "Hemodynamics"), the design of [power stations](https://en.wikipedia.org/wiki/Power_station "Power station"), the analysis of [pollution](https://en.wikipedia.org/wiki/Pollution "Pollution"), and many other problems. Coupled with [Maxwell's equations](https://en.wikipedia.org/wiki/Maxwell%27s_equations "Maxwell's equations"), they can be used to model and study [magnetohydrodynamics](https://en.wikipedia.org/wiki/Magnetohydrodynamics "Magnetohydrodynamics").

The Navier–Stokes equations are also of great interest in a purely mathematical sense. Despite their wide range of practical uses, it has not yet been proven whether smooth solutions always [exist](https://en.wikipedia.org/wiki/Existence_theorem "Existence theorem") in three dimensions—i.e., whether they are infinitely differentiable (or even just bounded) at all points in the [domain](https://en.wikipedia.org/wiki/Domain_\(mathematical_analysis\) "Domain (mathematical analysis)"). This is called the [Navier–Stokes existence and smoothness](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_existence_and_smoothness "Navier–Stokes existence and smoothness") problem. The [Clay Mathematics Institute](https://en.wikipedia.org/wiki/Clay_Mathematics_Institute "Clay Mathematics Institute") has called this one of the [seven most important open problems in mathematics](https://en.wikipedia.org/wiki/Millennium_Prize_Problems "Millennium Prize Problems") and has offered a [US$](https://en.wikipedia.org/wiki/US$ "US$")1 million prize for a solution or a counterexample.[[2]](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#cite_note-2)[[3]](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#cite_note-3)

(From https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations)

---

### Equation
Classical NS:
$$\frac{\partial u}{\partial t} + (u \cdot \nabla )u = -\nabla p+v \Delta u+f$$
>note $\nabla \cdot u = 0$

where:
$u$: velocity field
$p$: pressure
$v$: kinematic viscosity
$f$: external force
$\Delta$: Laplacian operator
$\nabla \cdot u = 0$: incompressibility condition

NS-Alpha (LANS-$\alpha$):
$$\frac{\partial v}{\partial t}+(u \cdot \nabla)v + (\nabla u)^{T} \cdot v = -\nabla p + v\Delta v+f$$
>note    $v =(1- \alpha^{2}\Delta)u$,
>	   $\nabla \cdot u = 0$

where:
$u$: filtered velocity field (large scales)
$v$: momentum velocity
$\alpha$: length scale parameter controlling the degree of smoothing

Differentiating the two:

| Aspect                | NS                                                        | NS-A                                                      |
| --------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| Purpose               | Governs all scales of incompressible fluid motion         | Regularized model for large-scale turbulent flow          |
| Small-Scale treatment | explicitly resolves all scales (difficult for turbulence) | Filters out small scales below length scale $\alpha$      |
| Mathematical Form     | Nonlinear PDE                                             | Modified nonlinear PDE with a smoothed velocity field     |
| Regularization        | None (singularities may form in 3D)                       | Regularized via Helmholtz operator ($1-\alpha^{2}\Delta$) |
| Computational Cost    | High for turbulent flows                                  | Lower for large eddy simulations (LES-like behavior)      |
The **Navier-Stokes-alpha equation** introduces a **filtered velocity field** to **regularize** the standard Navier-Stokes equations, making it more tractable for turbulent flow simulations. It suppresses the influence of small-scale eddies while preserving large-scale flow dynamics.

From ChatGPT (What is the Navier Stokes Alpha Equation, and how does it differ from the Navier Stokes Equation?)

---
![[Pasted image 20250529110020.png]]![[Pasted image 20250528120659.png]]
 from [[LaytonEtAl.pdf]]

rewritten:$$ u_{t}- \bar{u} \times (\nabla \times u) + \nabla q - v \Delta u =f$$
note:
$\nabla \cdot \bar{u}=0$
$-\alpha^{2}\Delta \bar{u}+\bar{u}=u$


 ---
$\nabla \cdot \underline{u} = 0$
is effectively just the conservation of mass
- note $\underline{u}$ represents our velocity, it has 3 components: $(u,v,w)$ for 3 dimensions
- recall $\nabla$ transforms $\underline{u}$ into $( \frac{\partial u}{\partial x}, \frac{\partial v}{\partial y}, \frac{\partial w}{\partial z} )$

$\rho \frac{D\underline{u} }{Dt} = -\nabla p + \mu \nabla^{2} \underline{u}+ \rho\underline{F}$
is effectively just $mass \times acceleration = Force$
- note that $-\nabla p$ and $\mu \nabla^{2} \underline{u}$ are our *internal* forces
	- $\nabla p$ represents our pressure gradient
	- $\mu \nabla^{2} \underline{u}$ is our viscosity
-  $\rho \underline{F}$ represents all our *external* forces (like gravity, electromagnetism, etc)



 from https://www.youtube.com/watch?v=ERBVFcutl3M

---
"Proving global regularity for the 3D Navier–Stokes equations (NSE) is one
of the most challenging outstanding problems in nonlinear analysis. The
main difficulty in establishing this result lies in controlling certain norms of
vorticity. More specifically, the vorticity stretching term in the 3D vorticity
equation forms the main obstacle to achieving this control."

"For more information and a brief guide to the previous litera-
ture specifically about the NS-a model, see paper \[20\]."


![[Pasted image 20250530170745.png]]

from https://www.math.uci.edu/~etiti/Publications/Holm/Foias/JDDE.pdf

This paper begins by reviewing a modeling scheme — called here the Navier–Stokes-alpha model, or NS-α model
(also called the viscous Camassa–Holm equations in [3–6]) — that introduces an energy “penalty” inhibiting the
creation of smaller and smaller excitations below a certain length scale (denoted alpha). This energy penalty results
in a nonlinearly dispersive modification of the Navier–Stokes equations. The alpha-modification appears in the
nonlinearity, it depends on length scale and we emphasize that it is dispersive, not dissipative

![[Pasted image 20250530172646.png]]
![[Pasted image 20250530175909.png]]
(2.7) is our model!

from [[TheNavierStokesAlphaModelOfFluidTurbulence.pdf]]


![[Pasted image 20250530173505.png]]

from Deepseek (tell me about the Navier-Stokes-Alpha equation)

---

### **Key Idea: Smoothing Out Turbulence**  
The **Navier-Stokes-Alpha (NS-α)** model modifies the standard **Navier-Stokes equations** to "blur" small, chaotic swirls in fluid motion while keeping large-scale movements intact. Think of it like applying a **softening filter** to a photo—tiny details get smoothed, but the overall structure remains clear.  

---

### **Breaking Down the Equations**  
The NS-α model consists of two main equations:  

1. **The Momentum Equation** (how velocity changes over time)  
2. **The Incompressibility Condition** (fluid doesn’t compress like a gas)  

Let’s examine them step by step.  

---

#### **1. Momentum Equation (Newton’s Second Law for Fluids)**  
The standard **Navier-Stokes momentum equation** is:  

$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p = \nu \nabla^2 \mathbf{u} + \mathbf{f}$  

**What each term means:**  

| Term | Name | Analogy (Calculus III Perspective) |  
|------|------|------------------------------------|  
| $\frac{\partial \mathbf{u}}{\partial t}$ | Time derivative of velocity | Measures how fast fluid velocity changes at a fixed point (like watching a river speed up over time). |  
| $(\mathbf{u} \cdot \nabla) \mathbf{u}$ | Nonlinear (convective) term | Represents fluid "pushing itself" (like a speeding car dragging nearby air along). |  
| $\nabla p$ | Pressure gradient | A "slope" in pressure pushing fluid from high to low pressure (like water flowing downhill). |  
| $\nu \nabla^2 \mathbf{u}$ | Viscous (diffusion) term | Friction slowing down nearby fluid layers (like honey resisting motion). |  
| $\mathbf{f}$ | External force | Wind, gravity, or pumps pushing the fluid. |  

**The NS-α Modification: Smoothing the Velocity**  
Instead of using the raw velocity $\mathbf{u}$, NS-α uses a **smoothed version** obtained via an **inverse Helmholtz filter**:  

$\mathbf{u} = (1 - \alpha^2 \nabla^2)^{-1} \tilde{\mathbf{u}}$  

**What does this mean?**  

- $\tilde{\mathbf{u}}$ = Original (unsmoothed) velocity field.  
- $\alpha$ = Smoothing scale (like a "blur radius"—bigger $\alpha$ means more smoothing).  
- $\nabla^2$ = Laplacian (from Calculus III: $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$).  
- $(1 - \alpha^2 \nabla^2)^{-1}$ = A mathematical operation that removes sharp, small-scale fluctuations.  

**Simpler Explanation:**  
This filter acts like a **weighted average** over nearby points, smoothing out tiny chaotic swirls while preserving large-scale flow patterns.  

---

#### **2. Incompressibility Condition (No Squishing Allowed)**  
The second equation enforces **mass conservation**:  

$\nabla \cdot \mathbf{u} = 0$  

**What it means:**  
- $\nabla \cdot \mathbf{u}$ is the **divergence** of velocity (from Calculus III).  
- If $\nabla \cdot \mathbf{u} = 0$, the fluid doesn’t compress or expand—like water flowing in a pipe with no leaks.  

**Why it matters:**  
This ensures that the fluid’s density remains constant, preventing unrealistic behavior.  

---

### **Putting It All Together**  
1. Start with the original **Navier-Stokes equations**.  
2. Replace the raw velocity $\tilde{\mathbf{u}}$ with a **smoothed version** $\mathbf{u}$ using the filter $(1 - \alpha^2 \nabla^2)^{-1}$.  
3. The smoothed $\mathbf{u}$ now appears in the momentum equation, reducing small-scale turbulence.  
4. Keep the **incompressibility condition** ($\nabla \cdot \mathbf{u} = 0$) to ensure mass conservation.  

**Final NS-α Equations (Simplified Form):**  
$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p = \nu \nabla^2 \mathbf{u} + \mathbf{f}$  
$\nabla \cdot \mathbf{u} = 0$  

---

### **Analogy: Traffic Flow Model**  
Imagine cars (fluid particles) on a highway:  
- **Standard Navier-Stokes** = Tracking every car’s exact speed (chaotic near traffic jams).  
- **NS-α Model** = Instead, looking at the **average speed over a stretch of road** (smoothing out sudden braking/accelerations).  
- $\alpha$ = How big of a stretch you average over (bigger $\alpha$ means smoother traffic reports).  

---

### **Summary for Calculus III Audience**  
- NS-α modifies Navier-Stokes by **smoothing velocity** to ignore tiny chaotic motions.  
- Uses a **filter** $(1 - \alpha^2 \nabla^2)^{-1}$ to blur small-scale turbulence.  
- Keeps the **incompressibility condition** ($\nabla \cdot \mathbf{u} = 0$) to conserve mass.  
- The math involves **divergence ($\nabla \cdot$)**, **gradient ($\nabla$)**, and **Laplacian ($\nabla^2$)** from multivariable calculus—no advanced differential equations required!  

from Deepseek (explain the mathematical Formulation section in much more detail. Make it accessible to someone who's highest level of math education is Calculus III, but not differential equations)

![[Pasted image 20250531212324.png]]

 from https://www.youtube.com/watch?v=QhjSwcSCeHQ