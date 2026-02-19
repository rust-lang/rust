// ignore-tidy-linelength
//@ revisions: sp disable enable-disable wrong-enable disable-enable-reset cfg-allow-first sp-allow-first
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future
//@ [sp] compile-flags:  -Z unstable-options -Z stack-protector=all
//@ [disable] compile-flags: -Z unstable-options -Z deny-partial-mitigations=stack-protector -Z stack-protector=all
//@ [enable-disable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z deny-partial-mitigations=stack-protector -Z stack-protector=all
//@ [wrong-enable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=control-flow-guard -Z stack-protector=all
//@ [cfg-allow-first] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -C control-flow-guard=on
//@ [sp-allow-first] compile-flags:  -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z stack-protector=all
//@ [disable-enable-reset] compile-flags: -Z unstable-options -Z deny-partial-mitigations=stack-protector -Z allow-partial-mitigations=stack-protector -Z stack-protector=all

fn main() {}
//~^ ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
//~| ERROR that is not compiled with
