// test for a union with a field that's a union with a manual impl Drop
// Ensures we do not treat all unions as not having any drop glue.

#![feature(untagged_unions)]

union Foo {
    bar: Bar, //~ ERROR unions cannot contain fields that may need dropping
}

union Bar {
    a: i32,
    b: u32,
}

impl Drop for Bar {
    fn drop(&mut self) {}
}

fn main() {}
