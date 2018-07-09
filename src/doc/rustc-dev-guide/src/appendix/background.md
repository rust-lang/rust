# Appendix B: Background topics

This section covers a numbers of common compiler terms that arise in
this guide. We try to give the general definition while providing some
Rust-specific context.

<a name="cfg"></a>

## What is a control-flow graph?

A control-flow graph is a common term from compilers. If you've ever
used a flow-chart, then the concept of a control-flow graph will be
pretty familiar to you. It's a representation of your program that
exposes the underlying control flow in a very clear way.

A control-flow graph is structured as a set of **basic blocks**
connected by edges. The key idea of a basic block is that it is a set
of statements that execute "together" – that is, whenever you branch
to a basic block, you start at the first statement and then execute
all the remainder. Only at the end of the block is there the
possibility of branching to more than one place (in MIR, we call that
final statement the **terminator**):

```mir
bb0: {
    statement0;
    statement1;
    statement2;
    ...
    terminator;
}
```

Many expressions that you are used to in Rust compile down to multiple
basic blocks. For example, consider an if statement:

```rust,ignore
a = 1;
if some_variable {
    b = 1;
} else {
    c = 1;
}
d = 1;
```

This would compile into four basic blocks:

```mir
BB0: {
    a = 1;
    if some_variable { goto BB1 } else { goto BB2 }
}

BB1: {
    b = 1;
    goto BB3;
}

BB2: {
    c = 1;
    goto BB3;
}

BB3: {
    d = 1;
    ...;
}
```

When using a control-flow graph, a loop simply appears as a cycle in
the graph, and the `break` keyword translates into a path out of that
cycle.

<a name="dataflow"></a>

## What is a dataflow analysis?

[*Static Program Analysis*](https://cs.au.dk/~amoeller/spa/) by Anders Møller
and Michael I. Schwartzbach is an incredible resource!

*to be written*

<a name="quantified"></a>

## What is "universally quantified"? What about "existentially quantified"?

*to be written*

<a name="variance"></a>

## What is co- and contra-variance?

Check out the subtyping chapter from the
[Rust Nomicon](https://doc.rust-lang.org/nomicon/subtyping.html).

See the [variance](./variance.html) chapter of this guide for more info on how
the type checker handles variance.

<a name="free-vs-bound"></a>

## What is a "free region" or a "free variable"? What about "bound region"?

Let's describe the concepts of free vs bound in terms of program
variables, since that's the thing we're most familiar with.

- Consider this expression, which creates a closure: `|a,
  b| a + b`. Here, the `a` and `b` in `a + b` refer to the arguments
  that the closure will be given when it is called. We say that the
  `a` and `b` there are **bound** to the closure, and that the closure
  signature `|a, b|` is a **binder** for the names `a` and `b`
  (because any references to `a` or `b` within refer to the variables
  that it introduces).
- Consider this expression: `a + b`. In this expression, `a` and `b`
  refer to local variables that are defined *outside* of the
  expression. We say that those variables **appear free** in the
  expression (i.e., they are **free**, not **bound** (tied up)).

So there you have it: a variable "appears free" in some
expression/statement/whatever if it refers to something defined
outside of that expressions/statement/whatever. Equivalently, we can
then refer to the "free variables" of an expression – which is just
the set of variables that "appear free".

So what does this have to do with regions? Well, we can apply the
analogous concept to type and regions. For example, in the type `&'a
u32`, `'a` appears free.  But in the type `for<'a> fn(&'a u32)`, it
does not.
