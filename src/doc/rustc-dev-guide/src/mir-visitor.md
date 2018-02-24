# MIR visitor

The MIR visitor is a convenient tool for traversing the MIR and either
looking for things or making changes to it. The visitor traits are
defined in [the `rustc::mir::visit` module][m-v] -- there are two of
them, generated via a single macro: `Visitor` (which operates on a
`&Mir` and gives back shared references) and `MutVisitor` (which
operates on a `&mut Mir` and gives back mutable references).

[m-v]: https://github.com/rust-lang/rust/tree/master/src/librustc/mir/visit.rs

To implement a visitor, you have to create a type that represents
your visitor. Typically, this type wants to "hang on" to whatever
state you will need while processing MIR:

```rust
struct MyVisitor<...> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
    ...
}
```

and you then implement the `Visitor` or `MutVisitor` trait for that type:

```rust
impl<'tcx> MutVisitor<'tcx> for NoLandingPads {
    fn visit_foo(&mut self, ...) {
        // ...
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

[`NoLandingPads`]: https://github.com/rust-lang/rust/tree/master/src/librustc_mir/transform/no_landing_pads.rs

