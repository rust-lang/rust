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

See the [variance](../variance.html) chapter of this guide for more info on how
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

# Further Reading About Compilers

> Thanks to `mem`, `scottmcm`, and `Levi` on the official Discord for the
> recommendations, and to `tinaun` for posting a link to a [twitter thread from
> Graydon Hoare](https://twitter.com/graydon_pub/status/1039615569132118016)
> which had some more recommendations!
> 
> Other sources: https://gcc.gnu.org/wiki/ListOfCompilerBooks
>
> If you have other suggestions, please feel free to open an issue or PR.

## Books
- [Types and Programming Languages](https://www.cis.upenn.edu/~bcpierce/tapl/)
- [Programming Language Pragmatics](https://www.cs.rochester.edu/~scott/pragmatics/)
- [Practical Foundations for Programming Languages](https://www.cs.cmu.edu/~rwh/pfpl/2nded.pdf)
- [Compilers: Principles, Techniques, and Tools, 2nd Edition](https://www.amazon.com/dp/9332518661/ref=cm_sw_r_other_apa_1tUSBb5VHAVA1)
- [Garbage Collection: Algorithms for Automatic Dynamic Memory Management](https://www.cs.kent.ac.uk/people/staff/rej/gcbook/)
- [Linkers and Loaders](https://linker.iecc.com/)
- [Advanced Compiler Design and Implementation](https://www.goodreads.com/book/show/887908.Advanced_Compiler_Design_and_Implementation)
- [Building an Optimizing Compiler](https://www.goodreads.com/book/show/2063103.Building_an_Optimizing_Compiler)
- [Crafting Interpreters](http://www.craftinginterpreters.com/)

## Courses
- [University of Oregon Programming Languages Summer School archive](https://www.cs.uoregon.edu/research/summerschool/archives.html)

## Wikis
- [Wikipedia](https://en.wikipedia.org/wiki/List_of_programming_languages_by_type)
- [Esoteric Programming Languages](https://esolangs.org/wiki/Main_Page)
- [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/index.html)
- [nLab](https://ncatlab.org/nlab/show/HomePage)

## Misc Papers and Blog Posts
- [Programming in Martin-Löf's Type Theory](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.6683&rep=rep1&type=pdf)
- [Polymorphism, Subtyping, and Type Inference in MLsub](https://www.cl.cam.ac.uk/~sd601/papers/mlsub-preprint.pdf)
