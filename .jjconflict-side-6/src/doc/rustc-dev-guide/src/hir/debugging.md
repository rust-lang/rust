# HIR Debugging

Use the `-Z unpretty=hir` flag to produce a human-readable representation of the HIR.
For cargo projects this can be done with `cargo rustc -- -Z unpretty=hir`.
This output is useful when you need to see at a glance how your code was desugared and transformed
during AST lowering.

For a full `Debug` dump of the data in the HIR, use the `-Z unpretty=hir-tree` flag.
This may be useful when you need to see the full structure of the HIR from the perspective of the
compiler.

If you are trying to correlate `NodeId`s or `DefId`s with source code, the
`-Z unpretty=expanded,identified` flag may be useful.

TODO: anything else? [#1159](https://github.com/rust-lang/rustc-dev-guide/issues/1159)
