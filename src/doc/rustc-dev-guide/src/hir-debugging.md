# HIR Debugging

The `-Z unpretty=hir-tree` flag will dump out the HIR.

If you are trying to correlate `NodeId`s or `DefId`s with source code, the
`--pretty expanded,identified` flag may be useful.

TODO: anything else? [#1159](https://github.com/rust-lang/rustc-dev-guide/issues/1159)
