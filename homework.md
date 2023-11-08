---
title: 💀 Домашка
order: 3
---

### Matrix calculus

1. Given a matrix $A$ of size $m \times n$ and a vector $x$ of size $n \times 1$, compute the gradient of the function $f(x) = \text{tr}(A^T A x x^T)$ with respect to $x$.

1. Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if $f(x) = \dfrac{1}{2} \Vert Ax - b\Vert^2_2$.
1. Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if 
	$$
	f(x) = \frac1m \sum\limits_{i=1}^m \log \left( 1 + \exp(a_i^{\top}x) \right) + \frac{\mu}{2}\Vert x\Vert _2^2, \; a_i, x \in \mathbb R^n, \; \mu>0
	$$
1. Compute the gradient $\nabla_A f(A)$ of the trace of the matrix exponential function $f(A) = \text{tr}(e^A)$ with respect to $A$. Hint: hint: Use the definition of the matrix exponential. Use the defintion of the differential $df = f(A + dA) - f(A) + o(\Vert dA \Vert)$ with the limit $\Vert dA \Vert \to 0$.
1. Find the gradient $\nabla f(x)$ and hessian $f''(x)$, if $f(x) = \frac{1}{2}\Vert A - xx^\top\Vert^2_F, A \in \mathbb{S}^n$
1. Calculate the first and the second derivative of the following function $f : S \to \mathbb{R}$
	$$
	f(t) = \text{det}(A − tI_n),
	$$
	where $A \in \mathbb{R}^{n \times n}, S := \{t \in \mathbb{R} : \text{det}(A − tI_n) \neq 0\}$.
1. Find the gradient $\nabla f(X)$, if $f(X) = \text{tr}\left( AX^2BX^{-\top} \right)$.

### Automatic differentiation and jax
You can use any automatic differentiation framework in this section (Jax, PyTorch, Autograd etc.)
1. You will work with the following function for this exercise,
	$$
	f(x,y)=e^{−\left(sin(x)−cos(y)\right)^2}
	$$	
	Draw the computational graph for the function. Note, that it should contain only primitive operations - you need to do it automatically -  [jax example](https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html), [PyTorch example](https://github.com/waleedka/hiddenlayer) - you can google/find your own way to visualise it.

1. Compare analytic and autograd (with any framework) approach for the calculation of the gradient of:		
	$$
	f(A) = \text{tr}(e^A)
	$$
1. We can use automatic differentiation not only to calculate necessary gradient, but also for tuning hyperparameters of the algorithm like learning rate in gradient descent (with gradient descent 🤯). Suppose, we have the following function $f(x) = \frac{1}{2}\Vert x\Vert^2$, select a random point $x_0 \in \mathbb{B}^{1000} = \{0 \leq x_i \leq 1 \mid \forall i\}$. Consider $10$ steps of the gradient descent starting from the point $x_0$:
	$$
	x_{k+1} = x_k - \alpha_k \nabla f(x_k)
	$$
	Your goal in this problem is to write the function, that takes $10$ scalar values $\alpha_i$ and return the result of the gradient descent on function $L = f(x_{10})$. And optimize this function using gradient descent on $\alpha \in \mathbb{R}^{10}$. Suppose that each of $10$ components of $\alpha$ is uniformly distributed on $[0; 0.1]$.
	$$
	\alpha_{k+1} = \alpha_k - \beta \frac{\partial L}{\partial \alpha}
	$$
	Choose any constant $\beta$ and the number of steps your need. Describe obtained results. How would you understand, that the obtained schedule ($\alpha \in \mathbb{R}^{10}$) becomes better, than it was at the start? How do you check numerically local optimality in this problem? 
1. Compare analytic and autograd (with any framework) approach for the gradient of:		
	$$
	f(X) = - \log \det X
	$$
1. Compare analytic and autograd (with any framework) approach for the gradient and hessian of:		
	$$
	f(x) = x^\top x x^\top x
	$$
---

### Convex sets

1. Show, that if $S \subseteq \mathbb{R}^n$ is convex set, then its interior $\mathbf{int } S$ and closure $\bar{S}$  are also convex sets.
1. Show, that $\mathbf{conv}\{xx^\top: x \in \mathbb{R}^n, \Vert x\Vert  = 1\} = \{A \in \mathbb{S}^n_+: \text{tr}(A) = 1\}$.
1. Let $K \subseteq \mathbb{R}^n_+$ is a cone. Prove that it is convex if and only if a set of $\{x \in K \mid \sum\limits_{i=1}^n x_i = 1 \}$ is convex.
1. Prove that the set of $\{x \in \mathbb{R}^2 \mid e^{x_1}\le x_2\}$ is convex.
1. Show that the set of directions of the non-strict local descending of the differentiable function in a point is a convex cone. (Previously, the question contained a typo "strict local descending")
1. Is the following set convex
	$$
	S = \left\{ a \in \mathbb{R}^k \mid p(0) = 1, \vert p(t) \vert\leq 1 \text{ for } \alpha\leq t \leq \beta\right\},
	$$
	where
	$$
	p(t) = a_1 + a_2 t + \ldots + a_k t^{k-1} \;?
	$$

---

### Convex functions

1. Consider the function $f(x) = x^d$, where $x \in \mathbb{R}_{+}$. Fill the following table with ✅ or ❎. Explain your answers

	| $d$ | Convex | Concave | Strictly Convex | $\mu$-strongly convex |
	|:-:|:-:|:-:|:-:|:-:|
	| $-2, x \in \mathbb{R}_{++}$| | | | |
	| $-1, x \in \mathbb{R}_{++}$| | | | |
	| $0$| | | | |
	| $0.5$ | | | | |
	|$1$ | | | | |
	| $\in (1; 2)$ | | | | |
	| $2$| | | | |
	| $> 2$| | | | 

	: {.responsive}

1. Prove that the entropy function, defined as

	$$
	f(x) = -\sum_{i=1}^n x_i \log(x_i),
	$$

	with $\text{dom}(f) = \{x \in \R^n_{++} : \sum_{i=1}^n x_i = 1\}$, is strictly concave.  

1. Show, that the function $f: \mathbb{R}^n_{++} \to \mathbb{R}$ is convex if $f(x) = - \prod\limits_{i=1}^n x_i^{\alpha_i}$ if $\mathbf{1}^T \alpha = 1, \alpha \succeq 0$.

1. Show that the maximum of a convex function $f$ over the polyhedron $P = \text{conv}\{v_1, \ldots, v_k\}$ is achieved at one of its vertices, i.e.,

	$$
	\sup_{x \in P} f(x) = \max_{i=1, \ldots, k} f(v_i).
	$$

	A stronger statement is: the maximum of a convex function over a closed bounded convex set is achieved at an extreme point, i.e., a point in the set that is not a convex combination of any other points in the set. (you do not have to prove it). *Hint:* Assume the statement is false, and use Jensen’s inequality.

1. Show, that the two definitions of $\mu$-strongly convex functions are equivalent:
	1. $f(x)$ is $\mu$-strongly convex $\iff$ for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$:
		
		$$
		f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \frac{\mu}{2} \lambda (1 - \lambda)\|x_1 - x_2\|^2
		$$

	1. $f(x)$ is $\mu$-strongly convex $\iff$ if there exists $\mu>0$ such that the function $f(x) - \dfrac{\mu}{2}\Vert x\Vert^2$ is convex.

### Conjugate sets
1. Let $\mathbb{A}_n$ be the set of all $n$ dimensional antisymmetric matrices (s.t. $X^T = - X$). Show that $\left( \mathbb{A}_n\right)^* = \mathbb{S}_n$. 
1. Find the sets $S^{*}, S^{**}, S^{***}$, if 
    
    $$
    S = \{ x \in \mathbb{R}^2 \mid x_1 + x_2 \ge 0, \;\; -\dfrac12x_1 + x_2 \ge 0, \;\; 2x_1 + x_2 \ge -1 \;\; -2x_1 + x_2 \ge -3\}
    $$

1. Prove, that $B_p$ and $B_{p_*}$ are inter-conjugate, i.e. $(B_p)^* = B_{p_*}, (B_{p_*})^* = B_p$, where $B_p$ is the unit ball (w.r.t. $p$ - norm) and $p, p_*$ are conjugated, i.e. $p^{-1} + p^{-1}_* = 1$. You can assume, that $p_* = \infty$ if $p = 1$ and vice versa.

---

### Conjugate functions

1. Find $f^*(y)$, if $f(x) = \vert 2x \vert$
1. Prove, that if $f(x) = \inf\limits_{u+v = x} (g(u) + h(v))$, then $f^*(y) = g^*(y) + h^*(y)$.
1. Find $f^*(y)$, if $f(x) = \log \left( \sum\limits_{i=1}^n e^{x_i} \right)$
1. Prove, that if $f(x) = g(Ax)$, then $f^*(y) = g^*(A^{-\top}y)$
1. Find $f^*(Y)$, if $f(X) = - \ln \det X, X \in \mathbb{S}^n_{++}$
1. The scalar Huber function is defined as

	$$
	f_{\text{hub}}(x) = 
	\begin{cases} 
	\frac{1}{2} x^2 & \text{if } |x| \leq 1 \\
	|x| - \frac{1}{2} & \text{if } |x| > 1
	\end{cases}
	$$

	![Scalar case](/huber_function.svg)

	This convex function arises in various applications, notably in robust estimation. This problem explores the generalizations of the Huber function to $\mathbb{R}^n$. A straightforward extension to $\mathbb{R}^n$ is expressed as $f_{\text{hub}}(x_1) + \ldots + f_{\text{hub}}(x_n)$, yet this formulation is not circularly symmetric, that is, it's not invariant under the transformation of $x$ by an orthogonal matrix. A circularly symmetric extension to $\mathbb{R}^n$ is given by

	$$
	f_{\text{cshub}}(x) = f_{\text{hub}}(\Vert x\Vert )= 
	\begin{cases} 
	\frac{1}{2} \Vert x\Vert_2 ^2 & \text{if } \Vert x\Vert_2 \leq 1 \\
	\Vert x\Vert_2 - \frac{1}{2} & \text{if } \Vert x\Vert_2 > 1
	\end{cases}
	$$

	where the subscript denotes "circularly symmetric Huber function". Show, that $f_{\text{cshub}}$ is convex. Find the conjugate function $f^*(y)$.

---

### Subgradient and subdifferential

1. Find $\partial f(x)$, if 
	$$
	f(x) = \text{Parametric ReLU}(x) = \begin{cases}
		x & \text{if } x > 0, \\
		ax & \text{otherwise}.
	\end{cases}
	$$
1. Prove, that $x_0$ - is the minimum point of a function $f(x)$ if and only if $0 \in \partial f(x_0)$.
1. Find $\partial f(x)$, if $f(x) = \Vert Ax - b\Vert _1$.
1. Find $\partial f(x)$, if $f(x) = e^{\Vert x\Vert}$.
1. Find $\partial f(x)$, if $f(x) = \frac12 \Vert Ax - b\Vert _2^2 + \lambda \Vert x\Vert_1, \quad \lambda > 0$.
1. Let $S \subseteq \mathbb{R}^n$ be a convex set. We will call a *normal cone* of the set $S$ at a point $x$ the following set:
	$$
    N_S(x) = \left\{c \in \mathbb{R}^n : \langle c, y-x\rangle \leq 0 \quad \forall y \in S\right\}
    $$
    i) Draw a normal cone for a set at the points $A, B, C, D, E, F$ on the figure below: 
	
		![Draw a normal cone for the set $S$ in these points](/normal_cone.svg)
    
	i) Show, that $N_S(x) = \{0\} \quad \forall x \in \mathbf{ri }(S)$.
    i) Show, that the subdifferential $\partial I_S(x) = N_S(x)$ if $I_S(x)$ is the indicator function, i.e. 
		$$
		I_S(x) = \begin{cases}0,\text{if } x \in S\\ \infty, \text{otherwise}\end{cases}
		$$

---

### KKT and duality

1. **Toy example**
	$$
	\begin{split}
	& x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
	\text{s.t. } & (x-2)(x-4) \leq 0
	\end{split}
	$$

	1. Give the feasible set, the optimal value, and the optimal solution.
	1.  Plot the objective $x^2 +1$ versus $x$. On the same plot, show the feasible set, optimal point and value, and plot the Lagrangian $L(x,\mu)$ versus $x$ for a few positive values of $\mu$. Verify the lower bound property ($p^* \geq \inf_x L(x, \mu)$for $\mu \geq 0$). Derive and sketch the Lagrange dual function $g$.
	1. State the dual problem, and verify that it is a concave maximization problem. Find the dual optimal value and dual optimal solution $\mu^*$. Does strong duality hold?
	1.  Let $p^*(u)$ denote the optimal value of the problem

	$$
	\begin{split}
	& x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
	\text{s.t. } & (x-2)(x-4) \leq u
	\end{split}
	$$

	as a function of the parameter $u$. Plot $p^*(u)$. Verify that $\dfrac{dp^*(0)}{du} = -\mu^*$ 

1. Derive the dual problem for the Ridge regression problem with $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^m, \lambda > 0$:

	$$
	\begin{split}
	\dfrac{1}{2}\|y-b\|^2 + \dfrac{\lambda}{2}\|x\|^2 &\to \min\limits_{x \in \mathbb{R}^n, y \in \mathbb{R}^m }\\
	\text{s.t. } & y = Ax
	\end{split}
	$$

1. Derive the dual problem for the support vector machine problem with $A \in \mathbb{R}^{m \times n}, \mathbf{1} \in \mathbb{R}^m \in \mathbb{R}^m, \lambda > 0$:

	$$
	\begin{split}
	\langle \mathbf{1}, t\rangle + \dfrac{\lambda}{2}\|x\|^2 &\to \min\limits_{x \in \mathbb{R}^n, t \in \mathbb{R}^m }\\
	\text{s.t. } & Ax \succeq \mathbf{1} - t \\
	& t \succeq 0
	\end{split}
	$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = 1, \\
	& x \succeq 0 
	\end{split}
	$$

	This problem can be considered as a simplest portfolio optimization problem.

1. Show, that the following problem has a unique solution and find it:

	$$
	\begin{split}
	& \langle C^{-1}, X\rangle - \log \det X \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & \langle Xa, a\rangle \leq 1,
	\end{split}
	$$

	where $C \in \mathbb{S}^n_{++}, a \in \mathbb{R}^n \neq 0$. The answer should not involve inversion of the matrix $C$.

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
	\end{split}
	$$

	where $A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$.

1.  Consider the equality constrained least-squares problem
	
	$$
	\begin{split}
	& \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Cx = d,
	\end{split}
	$$

	where $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = n$, and $C \in \mathbb{R}^{k \times n}$ with $\mathbf{rank }C = k$. Give the KKT conditions, and derive expressions for the primal solution $x^*$ and the dual solution $\lambda^*$.

1. Derive the KKT conditions for the problem
	
	$$
	\begin{split}
	& \mathbf{tr \;}X - \log\text{det }X \to \min\limits_{X \in \mathbb{S}^n_{++} }\\
	\text{s.t. } & Xs = y,
	\end{split}
	$$

	where $y \in \mathbb{R}^n$ and $s \in \mathbb{R}^n$ are given with $y^\top s = 1$. Verify that the optimal solution is given by

	$$
	X^* = I + yy^\top - \dfrac{1}{s^\top s}ss^\top
	$$

1.  **Supporting hyperplane interpretation of KKT conditions**. Consider a **convex** problem with no equality constraints
	
	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f_i(x) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Assume, that $\exists x^* \in \mathbb{R}^n, \mu^* \in \mathbb{R}^m$ satisfy the KKT conditions
	
	$$
	\begin{split}
    & \nabla_x L (x^*, \mu^*) = \nabla f_0(x^*) + \sum\limits_{i=1}^m\mu_i^*\nabla f_i(x^*) = 0 \\
    & \mu^*_i \geq 0, \quad i = [1,m] \\
    & \mu^*_i f_i(x^*) = 0, \quad i = [1,m]\\
    & f_i(x^*) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Show that

	$$
	\nabla f_0(x^*)^\top (x - x^*) \geq 0
	$$

	for all feasible $x$. In other words the KKT conditions imply the simple optimality criterion or $\nabla f_0(x^*)$ defines a supporting hyperplane to the feasible set at $x^*$.

1.  **Fenchel + Lagrange = ♥.** Express the dual problem of
	
	$$
	\begin{split}
	& c^\top x\to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f(x) \leq 0
	\end{split}
	$$

	with $c \neq 0$, in terms of the conjugate function $f^*$. Explain why the problem you give is convex. We do not assume $f$ is convex.
	
1. **A penalty method for equality constraints.** We consider the problem of minimization

	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax = b,
	\end{split}
	$$
	
	where $f_0(x): \mathbb{R}^n \to\mathbb{R} $ is convex and differentiable, and $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = m$. In a quadratic penalty method, we form an auxiliary function

	$$
	\phi(x) = f_0(x) + \alpha \|Ax - b\|_2^2,
	$$
	
	where $\alpha > 0$ is a parameter. This auxiliary function consists of the objective plus the penalty term $\alpha \Vert Ax - b\Vert_2^2$. The idea is that a minimizer of the auxiliary function, $\tilde{x}$, should be an approximate solution of the original problem. Intuition suggests that the larger the penalty weight $\alpha$, the better the approximation $\tilde{x}$ to a solution of the original problem. Suppose $\tilde{x}$ is a minimizer of $\phi(x)$. Show how to find, from $\tilde{x}$, a dual feasible point for the original problem. Find the corresponding lower bound on the optimal value of the original problem.
	
1. **Analytic centering.** Derive a dual problem for
	
	$$
	-\sum_{i=1}^m \log (b_i - a_i^\top x) \to \min\limits_{x \in \mathbb{R}^{n} }
	$$

	with domain $\{x \mid a^\top_i x < b_i , i = [1,m]\}$. 
	
	First introduce new variables $y_i$ and equality constraints $y_i = b_i − a^\top_i x$. (The solution of this problem is called the analytic center of the linear inequalities $a^\top_i x \leq b_i ,i = [1,m]$.  Analytic centers have geometric applications, and play an important role in barrier methods.) 
	