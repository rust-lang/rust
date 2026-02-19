// ignore-tidy-linelength
//@ revisions: no-deny deny-first
//@ check-pass
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition: 2024
//@ [deny-first] compile-flags: -Z deny-partial-mitigations=control-flow-guard -C control-flow-guard=on
//@ [no-deny] compile-flags: -C control-flow-guard=on

// check that the `-C control-flow-guard=on` overrides the `-Z deny-partial-mitigations=control-flow-guard`,
// which in edition 2024 leads to partial mitigations being allowed

fn main() {}
