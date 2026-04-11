// ignore-tidy-linelength
//@ revisions: stack-protector-future stack-protector-future-explicit-deny stack-protector-future-deny-reset-by-mitigation stack-protector-allow-then-deny stack-protector-but-allow-control-flow-guard control-flow-guard-future-allow-reset-by-mitigation stack-protector-future-allow-reset-by-mitigation stack-protector-future-deny-allow-reset-by-mitigation
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future

// msvc has an extra unwind dependency of std, normalize it in the error messages
//@ normalize-stderr: "\b(unwind|libc)\b" -> "unwind/libc"

// test that stack-protector is denied-partial in edition=future
//@ [stack-protector-future] compile-flags:  -Z unstable-options -Z stack-protector=all

// same, but with explicit deny
//@ [stack-protector-future-explicit-deny] compile-flags: -Z unstable-options -Z stack-protector=all -Z deny-partial-mitigations=stack-protector

// same, but with explicit deny before the enable. The `-Z stack-protector=all` resets the mitigation status
// to default which is deny at edition=future.
// at edition=2024, this would be allowed, see ok-allow-partial-mitigations-current-edition scenario stack-protector-future-deny-reset-by-mitigation
//@ [stack-protector-future-deny-reset-by-mitigation] compile-flags: -Z unstable-options -Z deny-partial-mitigations=stack-protector -Z stack-protector=all

// same, but with explicit allow followed by explicit deny
//@ [stack-protector-allow-then-deny] compile-flags: -Z unstable-options -Z stack-protector=all -Z allow-partial-mitigations=stack-protector -Z deny-partial-mitigations=stack-protector

// check that allowing an unrelated mitigation (control-flow-guard) does not allow a different mitigation (stack-protector)
//@ [stack-protector-but-allow-control-flow-guard] compile-flags: -Z unstable-options -Z stack-protector=all -Z allow-partial-mitigations=control-flow-guard

// check that `-C control-flow-guard` overrides the `-Z allow-partial-mitigations=control-flow-guard` (to the default, which is deny at edition=future)
//@ [control-flow-guard-future-allow-reset-by-mitigation] compile-flags: -Z unstable-options -Z allow-partial-mitigations=control-flow-guard -C control-flow-guard=on

// check that `-Z stack-protector` overrides the `-Z allow-partial-mitigations=stack-protector` (to the default, which is deny at edition=future)
//@ [stack-protector-future-allow-reset-by-mitigation] compile-flags:  -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z stack-protector=all

// check that this is the case even if there was a "deny" before the "allow"
//@ [stack-protector-future-deny-allow-reset-by-mitigation] compile-flags: -Z unstable-options -Z deny-partial-mitigations=stack-protector -Z allow-partial-mitigations=stack-protector -Z stack-protector=all

fn main() {}
//~^ ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
