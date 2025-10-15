// This test makes sure that the files used in the coverage are remapped by `--remap-path-prefix`
// and the `coverage` <- `object` scope.

//@ revisions: with_remap with_coverage_scope with_object_scope with_macro_scope
//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//
//@[with_coverage_scope] compile-flags: --remap-path-scope=coverage
//@[with_object_scope] compile-flags: --remap-path-scope=object
//@[with_macro_scope] compile-flags: --remap-path-scope=macro

fn main() {}
