// ignore-tidy-linelength
//@ revisions: control-flow-2024-explicit-deny
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition: 2024
//@ [control-flow-2024-explicit-deny] compile-flags: -C control-flow-guard=on -Z deny-partial-mitigations=control-flow-guard

// check that in edition 2024, it is still possible to explicitly
// disallow partial mitigations (in edition=future, they are
// disallowed by default)

fn main() {}
//~^ ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
