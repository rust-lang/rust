// ignore-tidy-linelength
//@ revisions: control-flow-guard-2024-default control-flow-guard-2024-deny-reset-by-mitigation stack-protector-2024-explicit-allow
//@ check-pass
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition: 2024

// check that the `-C control-flow-guard=on` overrides the `-Z deny-partial-mitigations=control-flow-guard`,
// which in edition 2024 leads to partial mitigations being allowed. Test with both an explicit
// deny and without one.

// just test control-flow-guard at edition 2024. allowed-partial due to backwards compatibility.
//@ [control-flow-guard-2024-default] compile-flags: -C control-flow-guard=on

// test that -C control-flow-guard=on resets -Z deny-partial-mitigations=control-flow-guard
//@ [control-flow-guard-2024-deny-reset-by-mitigation] compile-flags: -Z deny-partial-mitigations=control-flow-guard -C control-flow-guard=on

// also test that stack protector is fine in edition 2024 with an explicit allow
//@ [stack-protector-2024-explicit-allow] compile-flags: -Z stack-protector=all -Z allow-partial-mitigations=stack-protector

fn main() {}
