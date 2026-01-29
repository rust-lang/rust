// ignore-tidy-linelength
//@ revisions: sp both disable enable-disable wrong-enable enable-separately-disable-together
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future
//@ [both] compile-flags: -Z unstable-options -C control-flow-guard=on -Z stack-protector=all
//@ [sp] compile-flags:  -Z unstable-options -Z stack-protector=all
//@ [disable] compile-flags: -Z unstable-options -Z deny-partial-mitigations=stack-protector -Z stack-protector=all
//@ [enable-disable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z deny-partial-mitigations=stack-protector -Z stack-protector=all
//@ [wrong-enable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=control-flow-guard -Z stack-protector=all
//@ [enable-separately-disable-together] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z allow-partial-mitigations=control-flow-guard -Z deny-partial-mitigations=control-flow-guard,stack-protector -C control-flow-guard=on -Z stack-protector=all

fn main() {}
//[both]~^ ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[both]~| ERROR that is not compiled with
//[sp]~^^^^^^^^^^^ ERROR that is not compiled with
//[sp]~| ERROR that is not compiled with
//[sp]~| ERROR that is not compiled with
//[sp]~| ERROR that is not compiled with
//[sp]~| ERROR that is not compiled with
//[disable]~^^^^^^^^^^^^^^^^ ERROR that is not compiled with
//[disable]~| ERROR that is not compiled with
//[disable]~| ERROR that is not compiled with
//[disable]~| ERROR that is not compiled with
//[disable]~| ERROR that is not compiled with
//[enable-disable]~^^^^^^^^^^^^^^^^^^^^^ ERROR that is not compiled with
//[enable-disable]~| ERROR that is not compiled with
//[enable-disable]~| ERROR that is not compiled with
//[enable-disable]~| ERROR that is not compiled with
//[enable-disable]~| ERROR that is not compiled with
//[wrong-enable]~^^^^^^^^^^^^^^^^^^^^^^^^^^ ERROR that is not compiled with
//[wrong-enable]~| ERROR that is not compiled with
//[wrong-enable]~| ERROR that is not compiled with
//[wrong-enable]~| ERROR that is not compiled with
//[wrong-enable]~| ERROR that is not compiled with
//[enable-separately-disable-together]~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
//[enable-separately-disable-together]~| ERROR that is not compiled with
