//@ check-pass
//@ compile-flags: -Z track-diagnostics

struct A;
struct B;

pub const S: A = B;
