// ignore-tidy-linelength
//@ revisions: control-flow-guard-2024-default control-flow-guard-2024-deny-reset-by-mitigation stack-protector-2024-deny-reset-by-mitigation stack-protector-2024-allow-deny-reset-by-mitigation
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

// same but for stack-protector, to match the stack-protector-future-deny-reset-by-mitigation test in
// err-allow-partial-mitigations-1-error (which has the same args but on edition=future).
//@ [stack-protector-2024-deny-reset-by-mitigation] compile-flags: -Z deny-partial-mitigations=stack-protector -Z stack-protector=all

// check that this is the case even if there was an "allow" then a "deny"
//@ [stack-protector-2024-allow-deny-reset-by-mitigation] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z deny-partial-mitigations=stack-protector -Z stack-protector=all

fn main() {}
