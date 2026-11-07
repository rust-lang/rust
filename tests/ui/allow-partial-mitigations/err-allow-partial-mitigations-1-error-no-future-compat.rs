// ignore-tidy-file-linelength
//@ revisions: control-flow-guard-future-allow-reset-by-mitigation stack-protector-future-stable
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future

// msvc has an extra unwind dependency of std, normalize it in the error messages
//@ normalize-stderr: "\b(unwind|libc)\b" -> "unwind/libc"

// test for cases with no future compat warning, since the relevant '-C' option is passed.

// check that `-C control-flow-guard` overrides the `-Z allow-partial-mitigations=control-flow-guard` (to the default, which is deny at edition=future)
//@ [control-flow-guard-future-allow-reset-by-mitigation] compile-flags: -Z unstable-options -Z allow-partial-mitigations=control-flow-guard -C control-flow-guard=on

// same, but for `-C stack-protector`
//@ [stack-protector-future-stable] compile-flags: -Z unstable-options -C stack-protector=all

fn main() {}
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
//~? ERROR that is not compiled with
