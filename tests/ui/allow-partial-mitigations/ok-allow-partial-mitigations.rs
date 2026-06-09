// ignore-tidy-linelength
//@ revisions: stack-protector-explicit-allow stack-protector-and-control-flow-guard-explicit-allow stack-protector-deny-then-allow
//@ check-pass
//@ edition:future
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported

// requesting both stack-protector and control-flow-guard and then allow-partial-mitigations it
//@ [stack-protector-and-control-flow-guard-explicit-allow] compile-flags: -Z unstable-options -Z stack-protector=all -C control-flow-guard=on -Z allow-partial-mitigations=stack-protector,control-flow-guard

// requesting stack-protector and then allow-partial-mitigations it
//@ [stack-protector-explicit-allow] compile-flags:  -Z unstable-options -Z stack-protector=all -Z allow-partial-mitigations=stack-protector

// testing that the later allow-partial-mitigations overrides the earlier deny-partial-mitigations
// see also the stack-protector-allow-then-deny test (in the error tests) for the other order
//@ [stack-protector-deny-then-allow] compile-flags: -Z unstable-options -Z stack-protector=all -Z deny-partial-mitigations=stack-protector -Z allow-partial-mitigations=stack-protector

fn main() {}
