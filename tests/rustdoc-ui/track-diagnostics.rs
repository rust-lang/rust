//@ compile-flags: -Z track-diagnostics
//@ error-pattern: created at

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"

struct A;
struct B;

pub const S: A = B; //~ ERROR mismatched types
