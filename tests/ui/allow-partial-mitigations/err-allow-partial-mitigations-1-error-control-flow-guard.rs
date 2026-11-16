// ignore-tidy-file-linelength
//@ revisions: control-flow-guard-future-allow-reset-by-mitigation
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future

// msvc has an extra unwind dependency of std, normalize it in the error messages
//@ normalize-stderr: "\b(unwind|libc)\b" -> "unwind/libc"

// put the test for control-flow-guard in its own file since it does not have the future-compat warning,
// and you can't do negative revisions in compiletest

// check that `-C control-flow-guard` overrides the `-Z allow-partial-mitigations=control-flow-guard` (to the default, which is deny at edition=future)
//@ [control-flow-guard-future-allow-reset-by-mitigation] compile-flags: -Z unstable-options -Z allow-partial-mitigations=control-flow-guard -C control-flow-guard=on

fn main() {}
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
