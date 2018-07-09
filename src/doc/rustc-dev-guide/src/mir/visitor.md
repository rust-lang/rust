# MIR visitor

The MIR visitor is a convenient tool for traversing the MIR and either
looking for things or making changes to it. The visitor traits are
defined in [the `rustc::mir::visit` module][m-v] – there are two of
them, generated via a single macro: `Visitor` (which operates on a
`&Mir` and gives back shared references) and `MutVisitor` (which
operates on a `&mut Mir` and gives back mutable references).

[m-v]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/mir/visit/index.html

To implement a visitor, you have to create a type that represents
your visitor. Typically, this type wants to "hang on" to whatever
state you will need while processing MIR:

```rust,ignore
struct MyVisitor<...> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
    ...
}
```

and you then implement the `Visitor` or `MutVisitor` trait for that type:

```rust,ignore
impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
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

A very simple example of a visitor can be found in [`NoLandingPads`].
That visitor doesn't even require any state: it just visits all
terminators and removes their `unwind` successors.

[`NoLandingPads`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/no_landing_pads/struct.NoLandingPads.html

## Traversal

In addition the visitor, [the `rustc::mir::traversal` module][t]
contains useful functions for walking the MIR CFG in
[different standard orders][traversal] (e.g. pre-order, reverse
post-order, and so forth).

[t]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/mir/traversal/index.html
[traversal]: https://en.wikipedia.org/wiki/Tree_traversal

