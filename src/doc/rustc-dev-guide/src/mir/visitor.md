# MIR visitor

The MIR visitor is a convenient tool for traversing the MIR and either
looking for things or making changes to it. The visitor traits are
defined in [the `rustc_middle::mir::visit` module][m-v] – there are two of
them, generated via a single macro: `Visitor` (which operates on a
`&Mir` and gives back shared references) and `MutVisitor` (which
operates on a `&mut Mir` and gives back mutable references).

[m-v]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/visit/index.html

To implement a visitor, you have to create a type that represents
your visitor. Typically, this type wants to "hang on" to whatever
state you will need while processing MIR:

```rust,ignore
struct MyVisitor<...> {
    tcx: TyCtxt<'tcx>,
    ...
}
```

and you then implement the `Visitor` or `MutVisitor` trait for that type:

```rust,ignore
impl<'tcx> MutVisitor<'tcx> for MyVisitor {
    fn visit_foo(&mut self, ...) {
        ...
        self.super_foo(...);
    }
}
```

As shown above, within the impl, you can override any of the
`visit_foo` methods (e.g., `visit_terminator`) in order to write some
code that will execute whenever a `foo` is found. If you want to
recursively walk the contents of the `foo`, you then invoke the
`super_foo` method. (NB. You never want to override `super_foo`.)

A very simple example of a visitor can be found in [`LocalFinder`].
By implementing `visit_local` method, this visitor identifies local variables that
can be candidates for reordering.

[`LocalFinder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_transform/prettify/struct.LocalFinder.html

## Traversal

In addition the visitor, [the `rustc_middle::mir::traversal` module][t]
contains useful functions for walking the MIR CFG in
[different standard orders][traversal] (e.g. pre-order, reverse
post-order, and so forth).

[t]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/traversal/index.html
[traversal]: https://en.wikipedia.org/wiki/Tree_traversal

