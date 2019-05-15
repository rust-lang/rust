# An Overview of Chalk

> Chalk is under heavy development, so if any of these links are broken or if
> any of the information is inconsistent with the code or outdated, please
> [open an issue][rustc-issues] so we can fix it. If you are able to fix the
> issue yourself, we would love your contribution!

[Chalk][chalk] recasts Rust's trait system explicitly in terms of logic
programming by "lowering" Rust code into a kind of logic program we can then
execute queries against. (See [*Lowering to Logic*][lowering-to-logic] and
[*Lowering Rules*][lowering-rules]) Its goal is to be an executable, highly
readable specification of the Rust trait system.

There are many expected benefits from this work. It will consolidate our
existing, somewhat ad-hoc implementation into something far more principled and
expressive, which should behave better in corner cases, and be much easier to
extend.

## Chalk Structure

Chalk has two main "products". The first of these is the
[`chalk_engine`][chalk_engine] crate, which defines the core [SLG
solver][slg]. This is the part rustc uses.

The rest of chalk can be considered an elaborate testing harness. Chalk is
capable of parsing Rust-like "programs", lowering them to logic, and
performing queries on them.

Here's a sample session in the chalk repl, chalki. After feeding it our
program, we perform some queries on it.

```rust,ignore
?- program
Enter a program; press Ctrl-D when finished
| struct Foo { }
| struct Bar { }
| struct Vec<T> { }
| trait Clone { }
| impl<T> Clone for Vec<T> where T: Clone { }
| impl Clone for Foo { }

?- Vec<Foo>: Clone
Unique; substitution [], lifetime constraints []

?- Vec<Bar>: Clone
No possible solution.

?- exists<T> { Vec<T>: Clone }
Ambiguous; no inference guidance
```

You can see more examples of programs and queries in the [unit
tests][chalk-test-example].

Next we'll go through each stage required to produce the output above.

### Parsing ([chalk_parse])

Chalk is designed to be incorporated with the Rust compiler, so the syntax and
concepts it deals with heavily borrow from Rust. It is convenient for the sake
of testing to be able to run chalk on its own, so chalk includes a parser for a
Rust-like syntax. This syntax is orthogonal to the Rust AST and grammar. It is
not intended to look exactly like it or support the exact same syntax.

The parser takes that syntax and produces an [Abstract Syntax Tree (AST)][ast].
You can find the [complete definition of the AST][chalk-ast] in the source code.

The syntax contains things from Rust that we know and love, for example: traits,
impls, and struct definitions. Parsing is often the first "phase" of
transformation that a program goes through in order to become a format that
chalk can understand.

### Rust Intermediate Representation ([chalk_rust_ir])

After getting the AST we convert it to a more convenient intermediate
representation called [`chalk_rust_ir`][chalk_rust_ir]. This is sort of
analogous to the [HIR] in Rust. The process of converting to IR is called
*lowering*.

The [`chalk::program::Program`][chalk-program] struct contains some "rust things"
but indexed and accessible in a different way. For example, if you have a
type like `Foo<Bar>`, we would represent `Foo` as a string in the AST but in
`chalk::program::Program`, we use numeric indices (`ItemId`).

The [IR source code][ir-code] contains the complete definition.

### Chalk Intermediate Representation ([chalk_ir])

Once we have Rust IR it is time to convert it to "program clauses". A
[`ProgramClause`] is essentially one of the following:

* A [clause] of the form `consequence :- conditions` where `:-` is read as
  "if" and `conditions = cond1 && cond2 && ...`
* A universally quantified clause of the form
  `forall<T> { consequence :- conditions }`
  * `forall<T> { ... }` is used to represent [universal quantification]. See the
    section on [Lowering to logic][lowering-forall] for more information.
  * A key thing to note about `forall` is that we don't allow you to "quantify"
    over traits, only types and regions (lifetimes). That is, you can't make a
    rule like `forall<Trait> { u32: Trait }` which would say "`u32` implements
    all traits". You can however say `forall<T> { T: Trait }` meaning "`Trait`
    is implemented by all types".
  * `forall<T> { ... }` is represented in the code using the [`Binders<T>`
    struct][binders-struct].

*See also: [Goals and Clauses][goals-and-clauses]*

This is where we encode the rules of the trait system into logic. For
example, if we have the following Rust:

```rust,ignore
impl<T: Clone> Clone for Vec<T> {}
```

We generate the following program clause:

```rust,ignore
forall<T> { (Vec<T>: Clone) :- (T: Clone) }
```

This rule dictates that `Vec<T>: Clone` is only satisfied if `T: Clone` is also
satisfied (i.e. "provable").

Similar to [`chalk::program::Program`][chalk-program] which has "rust-like
things", chalk_ir defines [`ProgramEnvironment`] which is "pure logic".
The main field in that struct is `program_clauses`, which contains the
[`ProgramClause`]s generated by the rules module.

### Rules ([chalk_solve])

The `chalk_solve` crate ([source code][chalk_solve]) defines the logic rules we
use for each item in the Rust IR. It works by iterating over every trait, impl,
etc. and emitting the rules that come from each one.

*See also: [Lowering Rules][lowering-rules]*

#### Well-formedness checks

As part of lowering to logic, we also do some "well formedness" checks. See
the [`chalk_solve::wf` source code][solve-wf-src] for where those are done.

*See also: [Well-formedness checking][wf-checking]*

#### Coherence

The method `CoherenceSolver::specialization_priorities` in the `coherence` module
([source code][coherence-src]) checks "coherence", which means that it
ensures that two impls of the same trait for the same type cannot exist.

### Solver ([chalk_solve])

Finally, when we've collected all the program clauses we care about, we want
to perform queries on it. The component that finds the answer to these
queries is called the *solver*.

*See also: [The SLG Solver][slg]*

## Crates

Chalk's functionality is broken up into the following crates:
- [**chalk_engine**][chalk_engine]: Defines the core [SLG solver][slg].
- [**chalk_rust_ir**][chalk_rust_ir], containing the "HIR-like" form of the AST
- [**chalk_ir**][chalk_ir]: Defines chalk's internal representation of
  types, lifetimes, and goals.
- [**chalk_solve**][chalk_solve]: Combines `chalk_ir` and `chalk_engine`,
  effectively, which implements logic rules converting `chalk_rust_ir` to
  `chalk_ir`
  - Defines the `coherence` module, which implements coherence rules
  - [`chalk_engine::context`][engine-context] provides the necessary hooks.
- [**chalk_parse**][chalk_parse]: Defines the raw AST and a parser.
- [**chalk**][doc-chalk]: Brings everything together. Defines the following
  modules:
  - `chalk::lowering`, which converts AST to `chalk_rust_ir`
  - Also includes [chalki][chalki], chalk's REPL.

[Browse source code on GitHub](https://github.com/rust-lang/chalk)

## Testing

chalk has a test framework for lowering programs to logic, checking the
lowered logic, and performing queries on it. This is how we test the
implementation of chalk itself, and the viability of the [lowering
rules][lowering-rules].

The main kind of tests in chalk are **goal tests**. They contain a program,
which is expected to lower to logic successfully, and a set of queries
(goals) along with the expected output. Here's an
[example][chalk-test-example]. Since chalk's output can be quite long, goal
tests support specifying only a prefix of the output.

**Lowering tests** check the stages that occur before we can issue queries
to the solver: the [lowering to chalk_rust_ir][chalk-test-lowering], and the
[well-formedness checks][chalk-test-wf] that occur after that.

### Testing internals

Goal tests use a [`test!` macro][test-macro] that takes chalk's Rust-like
syntax and runs it through the full pipeline described above. The macro
ultimately calls the [`solve_goal` function][solve_goal].

Likewise, lowering tests use the [`lowering_success!` and
`lowering_error!` macros][test-lowering-macros].

## More Resources

* [Chalk Source Code](https://github.com/rust-lang/chalk)
* [Chalk Glossary](https://github.com/rust-lang/chalk/blob/master/GLOSSARY.md)

### Blog Posts

* [Lowering Rust traits to logic](http://smallcultfollowing.com/babysteps/blog/2017/01/26/lowering-rust-traits-to-logic/)
* [Unification in Chalk, part 1](http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/)
* [Unification in Chalk, part 2](http://smallcultfollowing.com/babysteps/blog/2017/04/23/unification-in-chalk-part-2/)
* [Negative reasoning in Chalk](http://aturon.github.io/blog/2017/04/24/negative-chalk/)
* [Query structure in chalk](http://smallcultfollowing.com/babysteps/blog/2017/05/25/query-structure-in-chalk/)
* [Cyclic queries in chalk](http://smallcultfollowing.com/babysteps/blog/2017/09/12/tabling-handling-cyclic-queries-in-chalk/)
* [An on-demand SLG solver for chalk](http://smallcultfollowing.com/babysteps/blog/2018/01/31/an-on-demand-slg-solver-for-chalk/)

[goals-and-clauses]: ./goals-and-clauses.html
[HIR]: ../hir.html
[lowering-forall]: ./lowering-to-logic.html#type-checking-generic-functions-beyond-horn-clauses
[lowering-rules]: ./lowering-rules.html
[lowering-to-logic]: ./lowering-to-logic.html
[slg]: ./slg.html
[wf-checking]: ./wf.html

[ast]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[chalk]: https://github.com/rust-lang/chalk
[rustc-issues]: https://github.com/rust-lang/rustc-guide/issues
[universal quantification]: https://en.wikipedia.org/wiki/Universal_quantification

[`ProgramClause`]: https://rust-lang.github.io/chalk/doc/chalk_ir/enum.ProgramClause.html
[`ProgramEnvironment`]: http://rust-lang.github.io/chalk/doc/chalk/program_environment/struct.ProgramEnvironment.html
[chalk_engine]: https://rust-lang.github.io/chalk/doc/chalk_engine/index.html
[chalk_ir]: https://rust-lang.github.io/chalk/doc/chalk_ir/index.html
[chalk_parse]: https://rust-lang.github.io/chalk/doc/chalk_parse/index.html
[chalk_solve]: https://rust-lang.github.io/chalk/doc/chalk_solve/index.html
[chalk_rust_ir]: https://rust-lang.github.io/chalk/doc/chalk_rust_ir/index.html
[doc-chalk]: https://rust-lang.github.io/chalk/doc/chalk/index.html
[engine-context]: https://rust-lang.github.io/chalk/doc/chalk_engine/context/index.html
[chalk-program]: http://rust-lang.github.io/chalk/doc/chalk/program/struct.Program.html

[binders-struct]: http://rust-lang.github.io/chalk/doc/chalk_ir/struct.Binders.html
[chalk-ast]: http://rust-lang.github.io/chalk/doc/chalk_parse/ast/index.html
[chalk-test-example]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/test.rs#L115
[chalk-test-lowering-example]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/rust_ir/lowering/test.rs#L8-L31
[chalk-test-lowering]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/rust_ir/lowering/test.rs
[chalk-test-wf]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/rules/wf/test.rs#L1
[chalki]: https://rust-lang.github.io/chalk/doc/chalki/index.html
[clause]: https://github.com/rust-lang/chalk/blob/master/GLOSSARY.md#clause
[coherence-src]: http://rust-lang.github.io/chalk/doc/chalk_solve/coherence/index.html
[ir-code]: http://rust-lang.github.io/chalk/doc/chalk_rust_ir/
[solve-wf-src]: http://rust-lang.github.io/chalk/doc/chalk_solve/wf/index.html
[solve_goal]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/test.rs#L85
[test-lowering-macros]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/test_util.rs#L21-L54
[test-macro]: https://github.com/rust-lang/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/test.rs#L33
