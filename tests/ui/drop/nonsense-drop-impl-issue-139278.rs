//@ check-fail
struct Foo;

impl Drop for Foo { //~ ERROR: not all trait items implemented
    const SPLOK: u32 = 0; //~ ERROR: not a member of trait
}

const X: Foo = Foo;

fn main() {}
