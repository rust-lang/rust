// This test checks the output of remapping with `--remap-path-prefix` and
// `--remap-path-scope` with a doctest.

//@ failure-status: 101
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

//@ revisions: with-diag-scope with-macro-scope with-object-scope with-doc-scope
//@ revisions: without-scope

//@ compile-flags:--test --test-args --test-threads=1
//@ compile-flags:-Z unstable-options --remap-path-prefix={{src-base}}=remapped_path

//@[with-diag-scope] compile-flags: -Zunstable-options --remap-path-scope=diagnostics
//@[with-macro-scope] compile-flags: -Zunstable-options --remap-path-scope=macro
//@[with-object-scope] compile-flags: -Zunstable-options --remap-path-scope=debuginfo
//@[with-doc-scope] compile-flags: -Zunstable-options --remap-path-scope=documentation

/// ```
/// fn invalid(
/// ```
pub struct SomeStruct;
