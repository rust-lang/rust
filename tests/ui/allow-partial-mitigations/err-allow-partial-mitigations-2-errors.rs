// ignore-tidy-linelength
//@ revisions: both enable-separately-disable-together enable-together-disable-separately
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future

// msvc has an extra unwind dependency of std, normalize it in the error messages
//@ normalize-stderr: "\b(unwind|libc)\b" -> "unwind/libc"

// just use 2 partial mitigations, without any allow/deny flag. Should be denied at edition=future.
//@ [both] compile-flags: -Z unstable-options -C control-flow-guard=on -Z stack-protector=all

// check that mitigations are denied if they are enabled separately and then disabled in a single command,
// to test the "foo,bar" syntax
//@ [enable-separately-disable-together] compile-flags: -Z unstable-options  -C control-flow-guard=on -Z stack-protector=all -Z allow-partial-mitigations=stack-protector -Z allow-partial-mitigations=control-flow-guard -Z deny-partial-mitigations=control-flow-guard,stack-protector

// same, but for allow
//@ [enable-together-disable-separately] compile-flags: -Z unstable-options  -C control-flow-guard=on -Z stack-protector=all -Z allow-partial-mitigations=stack-protector,control-flow-guard -Z deny-partial-mitigations=control-flow-guard -Z deny-partial-mitigations=stack-protector

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
