//@compile-flags: -Z track-diagnostics

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@normalize-stderr-test: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@normalize-stderr-test: "/rustc-dev/[0-9a-f]+/" -> ""

struct A;
struct B;
const S: A = B;
//~^ ERROR: mismatched types
//~| NOTE: created at

fn main() {}
