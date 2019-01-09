#![no_std]

extern crate std;

mod foo {
    pub fn test() -> Option<i32> {
        Some(2)
    }
}

fn main() {
    let a = core::option::Option::Some("foo");
    a.unwrap();
    foo::test().unwrap();
}
