//@ compile-flags: -Znext-solver
//@ check-pass

struct Foo(*mut ());

unsafe impl Sync for Foo {}

fn main() {}
