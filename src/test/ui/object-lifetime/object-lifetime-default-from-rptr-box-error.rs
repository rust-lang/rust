// Test that the lifetime from the enclosing `&` is "inherited"
// through the `Box` struct.

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct SomeStruct<'a> {
    t: &'a Box<dyn Test>,
}

fn c<'a>(t: &'a Box<dyn Test+'a>, mut ss: SomeStruct<'a>) {
    ss.t = t; //~ ERROR mismatched types
}

fn main() {
}
