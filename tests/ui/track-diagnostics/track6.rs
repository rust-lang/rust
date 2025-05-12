//@ compile-flags: -Z track-diagnostics
//@ error-pattern: created at

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"


pub trait Foo {
    fn bar();
}

impl <T> Foo for T {
    default fn bar() {} //~ ERROR specialization is unstable
}

fn main() {}
