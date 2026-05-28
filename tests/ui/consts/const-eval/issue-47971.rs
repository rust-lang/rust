//@ check-pass

struct S(pub &'static u32, pub u32);

const fn g(ss: &S) -> &u32 { &ss.1 }

static T: S = S(g(&T), 0);

fn main () { }
