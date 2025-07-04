//@ compile-flags: -Z track-diagnostics

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"

struct A;
struct B;

pub const S: A = B;
//~^ ERROR mismatched types
//~| NOTE created at
//~| NOTE expected `A`, found `B`
