// This test exercises `--remap-path-prefix` and `--remap-path-scope` with macros,
// like file!() and a diagnostic with compile_error!().
//
// See the compiler test suite for a more advanced tests, we just want to
// make sure here that rustdoc passes the right scopes to the underline rustc APIs.

//@ revisions: with-diag-scope with-macro-scope with-debuginfo-scope with-doc-scope
//@ revisions: without-scopes without-remap

//@[with-diag-scope] compile-flags: -Zunstable-options --remap-path-prefix={{src-base}}=remapped
//@[with-macro-scope] compile-flags: -Zunstable-options --remap-path-prefix={{src-base}}=remapped
//@[with-debuginfo-scope] compile-flags: -Zunstable-options --remap-path-prefix={{src-base}}=remapped
//@[with-doc-scope] compile-flags: -Zunstable-options --remap-path-prefix={{src-base}}=remapped
//@[without-scopes] compile-flags: -Zunstable-options --remap-path-prefix={{src-base}}=remapped

//@[with-diag-scope] compile-flags: -Zunstable-options --remap-path-scope=diagnostics
//@[with-macro-scope] compile-flags: -Zunstable-options --remap-path-scope=macro
//@[with-debuginfo-scope] compile-flags: -Zunstable-options --remap-path-scope=debuginfo
//@[with-doc-scope] compile-flags: -Zunstable-options --remap-path-scope=documentation

compile_error!(concat!("file!() = ", file!()));
//[with-macro-scope]~^ ERROR file!()
//[with-debuginfo-scope]~^^ ERROR file!()
//[with-doc-scope]~^^^ ERROR file!()
//[without-remap]~^^^^ ERROR file!()

//[with-diag-scope]~? ERROR file!()
//[without-scopes]~? ERROR file!()
