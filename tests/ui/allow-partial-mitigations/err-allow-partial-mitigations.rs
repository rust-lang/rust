// ignore-tidy-linelength
//@ revisions: sp both disable enable-disable wrong-enable
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future
//@ [both] compile-flags: -Z unstable-options -C control-flow-guard=on -Z stack-protector=all
//@ [sp] compile-flags:  -Z unstable-options -Z stack-protector=all
//@ [disable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=!stack-protector -Z stack-protector=all
//@ [enable-disable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=stack-protector -Z allow-partial-mitigations=!stack-protector -Z stack-protector=all
//@ [wrong-enable] compile-flags: -Z unstable-options -Z allow-partial-mitigations=control-flow-guard -Z stack-protector=all

fn main() {}
//[both]~^ ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[both]~| ERROR that is not protected by
//[sp]~^^^^^^^^^^^ ERROR that is not protected by
//[sp]~| ERROR that is not protected by
//[sp]~| ERROR that is not protected by
//[sp]~| ERROR that is not protected by
//[sp]~| ERROR that is not protected by
//[disable]~^^^^^^^^^^^^^^^^ ERROR that is not protected by
//[disable]~| ERROR that is not protected by
//[disable]~| ERROR that is not protected by
//[disable]~| ERROR that is not protected by
//[disable]~| ERROR that is not protected by
//[enable-disable]~^^^^^^^^^^^^^^^^^^^^^ ERROR that is not protected by
//[enable-disable]~| ERROR that is not protected by
//[enable-disable]~| ERROR that is not protected by
//[enable-disable]~| ERROR that is not protected by
//[enable-disable]~| ERROR that is not protected by
//[wrong-enable]~^^^^^^^^^^^^^^^^^^^^^^^^^^ ERROR that is not protected by
//[wrong-enable]~| ERROR that is not protected by
//[wrong-enable]~| ERROR that is not protected by
//[wrong-enable]~| ERROR that is not protected by
//[wrong-enable]~| ERROR that is not protected by
