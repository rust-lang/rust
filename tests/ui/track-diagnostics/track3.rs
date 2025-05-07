//@ compile-flags: -Z track-diagnostics
//@ error-pattern: created at

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"

fn main() {
    let _unimported = Blah { field: u8 };
    //~^ ERROR cannot find struct, variant or union type `Blah` in this scope
    //~| ERROR expected value, found builtin type `u8`
}
