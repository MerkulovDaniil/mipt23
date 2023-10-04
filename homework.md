---
title: 💀 Домашка
nav_order: 3
---

## Matrix calculus

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

	where $A \in \mathbb{R}^{n \times n}, S := \{t \in \mathbb{R} : \text{det}(A − tI_n) \neq 0\}	$.
1. Find the gradient $\nabla f(X)$, if $f(X) = \text{tr}\left( AX^2BX^{-\top} \right)$.

## Automatic differentiation and jax
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

## Convex sets

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

## Convex functions

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
	1. $f(x)$ is $\mu$-strongly convex $\iff$ $f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \mu \lambda (1 - \lambda)\|x_1 - x_2\|^2$ for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$.
	1. $f(x)$ is $\mu$-strongly convex $\iff$ if there exists $\mu>0$ such that the function $f(x) - \dfrac{\mu}{2}\Vert x\Vert^2$ is convex.

## Conjugate sets
1. Let $\mathbb{A}_n$ be the set of all $n$ dimensional antisymmetric matrices (s.t. $X^T = - X$). Show that $\left( \mathbb{A}_n\right)^* = \mathbb{S}_n$. 
1. Find the sets $S^{\star}, S^{\star\star}, S^{\star\star\star}$, if 
    
    $$
    S = \{ x \in \mathbb{R}^2 \mid x_1 + x_2 \ge 0, \;\; -\dfrac12x_1 + x_2 \ge 0, \;\; 2x_1 + x_2 \ge -1 \;\; -2x_1 + x_2 \ge -3\}
    $$

1. Prove, that $B_p$ and $B_{p_*}$ are inter-conjugate, i.e. $(B_p)^* = B_{p_*}, (B_{p_*})^* = B_p$, where $B_p$ is the unit ball (w.r.t. $p$ - norm) and $p, p_*$ are conjugated, i.e. $p^{-1} + p^{-1}_* = 1$. You can assume, that $p_* = \infty$ if $p = 1$ and vice versa.