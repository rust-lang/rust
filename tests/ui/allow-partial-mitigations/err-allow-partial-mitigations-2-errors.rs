// ignore-tidy-linelength
//@ revisions: both enable-separately-disable-together
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future
//@ [both] compile-flags: -Z unstable-options -C control-flow-guard=on -Z stack-protector=all
//@ [enable-separately-disable-together] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z allow-partial-mitigations=control-flow-guard -Z deny-partial-mitigations=control-flow-guard,stack-protector -C control-flow-guard=on -Z stack-protector=all

fn main() {}
//~^ ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
