# Type checking

The [`hir_analysis`] crate contains the source for "type collection" as well
as a bunch of related functionality.
Checking the bodies of functions is implemented in the [`hir_typeck`] crate.
These crates draw heavily on the [type inference] and [trait solving].

[`hir_analysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html
[`hir_typeck`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/index.html
[type inference]: ./type-inference.md
[trait solving]: ./traits/resolution.md

## Type collection

Type "collection" is the process of converting the types found in the HIR
(`hir::Ty`), which represent the syntactic things that the user wrote, into the
**internal representation** used by the compiler (`Ty<'tcx>`) – we also do
similar conversions for where-clauses and other bits of the function signature.

To try and get a sense of the difference, consider this function:

```rust,ignore
struct Foo { }
fn foo(x: Foo, y: self::Foo) { ... }
//        ^^^     ^^^^^^^^^
```

Those two parameters `x` and `y` each have the same type: but they will have
distinct `hir::Ty` nodes. Those nodes will have different spans, and of course
they encode the path somewhat differently. But once they are "collected" into
`Ty<'tcx>` nodes, they will be represented by the exact same internal type.

Collection is defined as a bundle of [queries] for computing information about
the various functions, traits, and other items in the crate being compiled.
Note that each of these queries is concerned with *interprocedural* things –
for example, for a function definition, collection will figure out the type and
signature of the function, but it will not visit the *body* of the function in
any way, nor examine type annotations on local variables (that's the job of
type *checking*).

For more details, see the [`collect`][collect] module.

[queries]: ./query.md
[collect]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/collect/index.html

**TODO**: actually talk about type checking... [#1161](https://github.com/rust-lang/rustc-dev-guide/issues/1161)
