//@compile-flags: -Z track-diagnostics

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@normalize-stderr-test: ".rs:\d+:\d+" -> ".rs:LL:CC"

struct A;
struct B;
const S: A = B;

fn main() {}
