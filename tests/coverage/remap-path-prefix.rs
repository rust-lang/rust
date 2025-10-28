// This test makes sure that the files used in the coverage are remapped by
// `--remap-path-prefix` and the `coverage` <- `object` scopes.
//
// We also test the `macro` scope to make sure it does not affect coverage.

// When coverage paths are remapped, the coverage-run mode can't find source files (because
// it doesn't know about the remapping), so it produces an empty coverage report. The empty
// report (i.e. no `.coverage` files) helps to demonstrate that remapping was indeed performed.

//@ revisions: with_remap with_coverage_scope with_object_scope with_macro_scope
//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//
//@[with_coverage_scope] compile-flags: -Zremap-path-scope=coverage
//@[with_object_scope] compile-flags: -Zremap-path-scope=object
//@[with_macro_scope] compile-flags: -Zremap-path-scope=macro

fn main() {}
