// ignore-tidy-linelength
//@ revisions: sp both disable-enable
//@ check-pass
//@ edition:future
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ [both] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector,control-flow-guard -C control-flow-guard=on -Z stack-protector=all
//@ [sp] compile-flags:  -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z stack-protector=all
//@ [disable-enable] compile-flags: -Z unstable-options -Z deny-partial-mitigations=stack-protector -Z allow-partial-mitigations=stack-protector -Z stack-protector=all

fn main() {}
