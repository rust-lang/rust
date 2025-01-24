# Background topics

This section covers a numbers of common compiler terms that arise in
this guide. We try to give the general definition while providing some
Rust-specific context.

<a id="cfg"></a>

## What is a control-flow graph?

A control-flow graph (CFG) is a common term from compilers. If you've ever
used a flow-chart, then the concept of a control-flow graph will be
pretty familiar to you. It's a representation of your program that
clearly exposes the underlying control flow.

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

This would compile into four basic blocks in MIR. In textual form, it looks like
this:

```mir
BB0: {
    a = 1;
    if some_variable {
        goto BB1;
    } else {
        goto BB2;
    }
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
    ...
}
```

In graphical form, it looks like this:

```
                BB0
       +--------------------+
       | a = 1;             |
       +--------------------+
             /       \
  if some_variable   else
           /           \
     BB1  /             \  BB2
    +-----------+   +-----------+
    | b = 1;    |   | c = 1;    |
    +-----------+   +-----------+
            \          /
             \        /
              \ BB3  /
            +----------+
            | d = 1;   |
            | ...      |
            +----------+
```

When using a control-flow graph, a loop simply appears as a cycle in
the graph, and the `break` keyword translates into a path out of that
cycle.

<a id="dataflow"></a>

## What is a dataflow analysis?

[*Static Program Analysis*](https://cs.au.dk/~amoeller/spa/) by Anders Møller
and Michael I. Schwartzbach is an incredible resource!

_Dataflow analysis_ is a type of static analysis that is common in many
compilers. It describes a general technique, rather than a particular analysis.

The basic idea is that we can walk over a [control-flow graph (CFG)](#cfg) and
keep track of what some value could be. At the end of the walk, we might have
shown that some claim is true or not necessarily true (e.g. "this variable must
be initialized"). `rustc` tends to do dataflow analyses over the MIR, since MIR
is already a CFG.

For example, suppose we want to check that `x` is initialized before it is used
in this snippet:

```rust,ignore
fn foo() {
    let mut x;

    if some_cond {
        x = 1;
    }

    dbg!(x);
}
```

A CFG for this code might look like this:

```txt
 +------+
 | Init | (A)
 +------+
    |   \
    |   if some_cond
  else    \ +-------+
    |      \| x = 1 | (B)
    |       +-------+
    |      /
 +---------+
 | dbg!(x) | (C)
 +---------+
```

We can do the dataflow analysis as follows: we will start off with a flag `init`
which indicates if we know `x` is initialized. As we walk the CFG, we will
update the flag. At the end, we can check its value.

So first, in block (A), the variable `x` is declared but not initialized, so
`init = false`. In block (B), we initialize the value, so we know that `x` is
initialized. So at the end of (B), `init = true`.

Block (C) is where things get interesting. Notice that there are two incoming
edges, one from (A) and one from (B), corresponding to whether `some_cond` is true or not.
But we cannot know that! It could be the case the `some_cond` is always true,
so that `x` is actually always initialized. It could also be the case that
`some_cond` depends on something random (e.g. the time), so `x` may not be
initialized. In general, we cannot know statically (due to [Rice's
Theorem][rice]).  So what should the value of `init` be in block (C)?

[rice]: https://en.wikipedia.org/wiki/Rice%27s_theorem

Generally, in dataflow analyses, if a block has multiple parents (like (C) in
our example), its dataflow value will be some function of all its parents (and
of course, what happens in (C)).  Which function we use depends on the analysis
we are doing.

In this case, we want to be able to prove definitively that `x` must be
initialized before use. This forces us to be conservative and assume that
`some_cond` might be false sometimes. So our "merging function" is "and". That
is, `init = true` in (C) if `init = true` in (A) _and_ in (B) (or if `x` is
initialized in (C)). But this is not the case; in particular, `init = false` in
(A), and `x` is not initialized in (C).  Thus, `init = false` in (C); we can
report an error that "`x` may not be initialized before use".

There is definitely a lot more that can be said about dataflow analyses. There is an
extensive body of research literature on the topic, including a lot of theory.
We only discussed a forwards analysis, but backwards dataflow analysis is also
useful. For example, rather than starting from block (A) and moving forwards,
we might have started with the usage of `x` and moved backwards to try to find
its initialization.

<a id="quantified"></a>

## What is "universally quantified"? What about "existentially quantified"?

In math, a predicate may be _universally quantified_ or _existentially
quantified_:

- _Universal_ quantification:
  - the predicate holds if it is true for all possible inputs.
  - Traditional notation: ∀x: P(x). Read as "for all x, P(x) holds".
- _Existential_ quantification:
  - the predicate holds if there is any input where it is true, i.e., there
    only has to be a single input.
  - Traditional notation: ∃x: P(x). Read as "there exists x such that P(x) holds".

In Rust, they come up in type checking and trait solving. For example,

```rust,ignore
fn foo<T>()
```
This function claims that the function is well-typed for all types `T`: `∀ T: well_typed(foo)`.

Another example:

```rust,ignore
fn foo<'a>(_: &'a usize)
```
This function claims that for any lifetime `'a` (determined by the
caller), it is well-typed: `∀ 'a: well_typed(foo)`.

Another example:

```rust,ignore
fn foo<F>()
where for<'a> F: Fn(&'a u8)
```
This function claims that it is well-typed for all types `F` such that for all
lifetimes `'a`, `F: Fn(&'a u8)`: `∀ F: ∀ 'a: (F: Fn(&'a u8)) => well_typed(foo)`.

One more example:

```rust,ignore
fn foo(_: dyn Debug)
```
This function claims that there exists some type `T` that implements `Debug`
such that the function is well-typed: `∃ T:  (T: Debug) and well_typed(foo)`.

<a id="variance"></a>

## What is a de Bruijn Index?

[De Bruijn indices][wikideb] are a way of representing, using only integers,
which variables are bound in which binders. They were originally invented for
use in lambda calculus evaluation (see [this Wikipedia article][wikideb] for
more). In `rustc`, we use de Bruijn indices to [represent generic types][sub].

[wikideb]: https://en.wikipedia.org/wiki/De_Bruijn_index
[sub]: ../ty_module/generic_arguments.md


Here is a basic example of how de Bruijn indices might be used for closures (we
don't actually do this in `rustc` though!):

```rust,ignore
|x| {
    f(x) // de Bruijn index of `x` is 1 because `x` is bound 1 level up

    |y| {
        g(x, y) // index of `x` is 2 because it is bound 2 levels up
                // index of `y` is 1 because it is bound 1 level up
    }
}
```

## What are co- and contra-variance?

Check out the subtyping chapter from the
[Rust Nomicon](https://doc.rust-lang.org/nomicon/subtyping.html).

See the [variance](../variance.html) chapter of this guide for more info on how
the type checker handles variance.

<a id="free-vs-bound"></a>

## What is a "free region" or a "free variable"? What about "bound region"?

Let's describe the concepts of free vs bound in terms of program
variables, since that's the thing we're most familiar with.

- Consider this expression, which creates a closure: `|a, b| a + b`.
  Here, the `a` and `b` in `a + b` refer to the arguments that the closure will
  be given when it is called. We say that the `a` and `b` there are **bound** to
  the closure, and that the closure signature `|a, b|` is a **binder** for the
  names `a` and `b` (because any references to `a` or `b` within refer to the
  variables that it introduces).
- Consider this expression: `a + b`. In this expression, `a` and `b` refer to
  local variables that are defined *outside* of the expression. We say that
  those variables **appear free** in the expression (i.e., they are **free**,
  not **bound** (tied up)).

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
> Graydon Hoare](https://web.archive.org/web/20181230012554/https://twitter.com/graydon_pub/status/1039615569132118016)
> which had some more recommendations!
>
> Other sources: https://gcc.gnu.org/wiki/ListOfCompilerBooks
>
> If you have other suggestions, please feel free to open an issue or PR.

## Books
- [Types and Programming Languages](https://www.cis.upenn.edu/~bcpierce/tapl/)
- [Programming Language Pragmatics](https://www.cs.rochester.edu/~scott/pragmatics/)
- [Practical Foundations for Programming Languages](https://www.cs.cmu.edu/~rwh/pfpl/)
- [Compilers: Principles, Techniques, and Tools, 2nd Edition](https://www.pearson.com/us/higher-education/program/Aho-Compilers-Principles-Techniques-and-Tools-2nd-Edition/PGM167067.html)
- [Garbage Collection: Algorithms for Automatic Dynamic Memory Management](https://www.cs.kent.ac.uk/people/staff/rej/gcbook/)
- [Linkers and Loaders](https://www.amazon.com/Linkers-Kaufmann-Software-Engineering-Programming/dp/1558604960) (There are also free versions of this, but the version we had linked seems to be offline at the moment.)
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
- [Programming in Martin-Löf's Type Theory](https://www.cse.chalmers.se/research/group/logic/book/)
- [Polymorphism, Subtyping, and Type Inference in MLsub](https://dl.acm.org/doi/10.1145/3093333.3009882)
