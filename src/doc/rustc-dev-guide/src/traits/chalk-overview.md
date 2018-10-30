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
[`chalk_engine`][doc-chalk-engine] crate, which defines the core [SLG
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

You can see more examples of programs and queries in the [unit tests][chalk-tests].

[chalk-tests]: https://github.com/rust-lang-nursery/chalk/blob/4bce000801de31bf45c02f742a5fce335c9f035f/src/test.rs#L115

### Crates
- [**chalk_engine**][doc-chalk-engine]: Defines the core [SLG solver][slg].
- [**chalk_ir**][doc-chalk-ir]: Defines chalk's internal representation of
  types, lifetimes, and goals.
- [**chalk_solve**][doc-chalk-solve]: Combines `chalk_ir` and `chalk_engine`,
  effectively.
  - [`chalk_engine::context`][engine-context] provides the necessary hooks.
- [**chalk_parse**][doc-chalk-parse]: Defines the raw AST and a parser.
- [**chalk**][doc-chalk]: Brings everything together. Defines the following
  modules:
  - [`rust_ir`][doc-chalk-rust-ir], containing the "HIR-like" form of the AST
    - `rust_ir::lowering`, which converts AST to `rust_ir`
  - `rules`, which implements logic rules
    converting `rust_ir` to `chalk_ir`
  - `coherence`, which implements coherence rules
  - Also includes [chalki][doc-chalki], chalk's REPL.

[Browse source on GitHub](https://github.com/rust-lang-nursery/chalk)

[engine-context]: https://rust-lang-nursery.github.io/chalk/doc/chalk_engine/context/index.html

[doc-chalk-engine]: https://rust-lang-nursery.github.io/chalk/doc/chalk_engine/index.html
[doc-chalk-ir]: https://rust-lang-nursery.github.io/chalk/doc/chalk_ir/index.html
[doc-chalk-solve]: https://rust-lang-nursery.github.io/chalk/doc/chalk_solve/index.html
[doc-chalk-parse]: https://rust-lang-nursery.github.io/chalk/doc/chalk_parse/index.html
[doc-chalk]: https://rust-lang-nursery.github.io/chalk/doc/chalk/index.html
[doc-chalk-rust-ir]: https://rust-lang-nursery.github.io/chalk/doc/chalk/rules/index.html
[doc-chalki]: https://rust-lang-nursery.github.io/chalk/doc/chalki/index.html

## Resources

* [Chalk Source Code](https://github.com/rust-lang-nursery/chalk)
* [Chalk Glossary](https://github.com/rust-lang-nursery/chalk/blob/master/GLOSSARY.md)
* The traits section of the rustc guide (you are here)

### Blog Posts

* [Lowering Rust traits to logic](http://smallcultfollowing.com/babysteps/blog/2017/01/26/lowering-rust-traits-to-logic/)
* [Unification in Chalk, part 1](http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/)
* [Unification in Chalk, part 2](http://smallcultfollowing.com/babysteps/blog/2017/04/23/unification-in-chalk-part-2/)
* [Negative reasoning in Chalk](http://aturon.github.io/blog/2017/04/24/negative-chalk/)
* [Query structure in chalk](http://smallcultfollowing.com/babysteps/blog/2017/05/25/query-structure-in-chalk/)
* [Cyclic queries in chalk](http://smallcultfollowing.com/babysteps/blog/2017/09/12/tabling-handling-cyclic-queries-in-chalk/)
* [An on-demand SLG solver for chalk](http://smallcultfollowing.com/babysteps/blog/2018/01/31/an-on-demand-slg-solver-for-chalk/)

## Parsing

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

## Lowering

After parsing, there is a "lowering" phase. This aims to convert traits/impls
into "program clauses". A [`ProgramClause` (source code)][programclause] is
essentially one of the following:

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

Lowering is the phase where we encode the rules of the trait system into logic.
For example, if we have the following Rust:

```rust,ignore
impl<T: Clone> Clone for Vec<T> {}
```

We generate the following program clause:

```rust,ignore
forall<T> { (Vec<T>: Clone) :- (T: Clone) }
```

This rule dictates that `Vec<T>: Clone` is only satisfied if `T: Clone` is also
satisfied (i.e. "provable").

### Well-formedness checks

As part of lowering from the AST to the internal IR, we also do some "well
formedness" checks. See the [source code][well-formedness-checks] for where
those are done. The call to `record_specialization_priorities` checks
"coherence" which means that it ensures that two impls of the same trait for the
same type cannot exist.

## Intermediate Representation (IR)

The second intermediate representation in chalk is called, well, the "ir". :)
The [IR source code][ir-code] contains the complete definition. The
`ir::Program` struct contains some "rust things" but indexed and accessible in
a different way. This is sort of analogous to the [HIR] in Rust.

For example, if you have a type like `Foo<Bar>`, we would represent `Foo` as a
string in the AST but in `ir::Program`, we use numeric indices (`ItemId`).

In addition to `ir::Program` which has "rust-like things", there is also
`ir::ProgramEnvironment` which is "pure logic". The main field in that struct is
`program_clauses` which contains the `ProgramClause`s that we generated
previously.

## Rules

The `rules` module works by iterating over every trait, impl, etc. and emitting
the rules that come from each one. See [Lowering Rules][lowering-rules] for the
most up-to-date reference on that.

The `ir::ProgramEnvironment` is created [in this module][rules-environment].

## Testing

TODO: Basically, [there is a macro](https://github.com/rust-lang-nursery/chalk/blob/94a1941a021842a5fcb35cd043145c8faae59f08/src/solve/test.rs#L112-L148)
that will take chalk's Rust-like syntax and run it through the full pipeline
described above.
[This](https://github.com/rust-lang-nursery/chalk/blob/94a1941a021842a5fcb35cd043145c8faae59f08/src/solve/test.rs#L83-L110)
is the function that is ultimately called.

## Solver

See [The SLG Solver][slg].

[rustc-issues]: https://github.com/rust-lang-nursery/rustc-guide/issues
[chalk]: https://github.com/rust-lang-nursery/chalk
[lowering-to-logic]: ./lowering-to-logic.html
[lowering-rules]: ./lowering-rules.html
[ast]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[chalk-ast]: https://github.com/rust-lang-nursery/chalk/blob/master/chalk-parse/src/ast.rs
[universal quantification]: https://en.wikipedia.org/wiki/Universal_quantification
[lowering-forall]: ./lowering-to-logic.html#type-checking-generic-functions-beyond-horn-clauses
[programclause]: https://github.com/rust-lang-nursery/chalk/blob/94a1941a021842a5fcb35cd043145c8faae59f08/src/ir.rs#L721
[clause]: https://github.com/rust-lang-nursery/chalk/blob/master/GLOSSARY.md#clause
[goals-and-clauses]: ./goals-and-clauses.html
[well-formedness-checks]: https://github.com/rust-lang-nursery/chalk/blob/94a1941a021842a5fcb35cd043145c8faae59f08/src/ir/lowering.rs#L230-L232
[ir-code]: https://github.com/rust-lang-nursery/chalk/tree/master/chalk-ir
[HIR]: ../hir.html
[binders-struct]: https://github.com/rust-lang-nursery/chalk/blob/94a1941a021842a5fcb35cd043145c8faae59f08/src/ir.rs#L661
[rules-environment]: https://github.com/rust-lang-nursery/chalk/blob/94a1941a021842a5fcb35cd043145c8faae59f08/src/rules.rs#L9
[slg]: ./slg.html
