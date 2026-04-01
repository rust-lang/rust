//@ check-pass
#![allow(unused_assignments)]
#![allow(unused_variables)]
// Test that a field can have the same name in different variants
// of an enum

pub enum Foo {
    X { foo: u32 },
    Y { foo: u32 }
}

pub fn foo(mut x: Foo) {
    let mut y = None;
    let mut z = None;
    if let Foo::X { ref foo } = x {
        z = Some(foo);
    }
    if let Foo::Y { ref mut foo } = x {
        y = Some(foo);
    }
}

fn main() {}
