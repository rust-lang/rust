// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![feature(untagged_unions)]

use std::ops::AddAssign;

struct Dropping;
impl AddAssign for Dropping {
    fn add_assign(&mut self, _: Self) {}
}

union Foo {
    a: u8, // non-dropping
    b: Dropping, // treated as dropping
}

fn main() {
    let mut foo = Foo { a: 42 };
    foo.a += 5; //~ ERROR access to union field is unsafe
    foo.b += Dropping; //~ ERROR access to union field is unsafe
    foo.b = Dropping; //~ ERROR assignment to union field that might need dropping is unsafe
    foo.a; //~ ERROR access to union field is unsafe
    let foo = Foo { a: 42 };
    foo.b; //~ ERROR access to union field is unsafe
    let mut foo = Foo { a: 42 };
    foo.b = foo.b;
    //~^ ERROR access to union field is unsafe
    //~| ERROR assignment to union field that might need dropping
}
