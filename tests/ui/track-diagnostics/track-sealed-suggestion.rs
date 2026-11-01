//@ compile-flags: -Z track-diagnostics
//@ dont-require-annotations: NOTE

//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@ normalize-stderr: "/rustc(?:-dev)?/[a-z0-9.]+/" -> ""

strct Human;
//~^ ERROR expected one of
//~| NOTE created at

fn main() {}
