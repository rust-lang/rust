#![crate_type = "lib"]
#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

enum Foo {
    Bar = 42,
    Baz,
    Quux = 100,
}

#[miri_run]
fn foo() -> [u8; 3] {
    [Foo::Bar as u8, Foo::Baz as u8, Foo::Quux as u8]
}

#[miri_run]
fn unsafe_match() -> bool {
    match unsafe { std::mem::transmute::<u8, Foo>(43) } {
        Foo::Baz => true,
        _ => false,
    }
}
