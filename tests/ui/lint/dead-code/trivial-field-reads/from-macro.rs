//@ aux-crate: macro_def=macro_def.rs

#![feature(rustc_attrs)]
#![deny(dead_code)]

#[derive(Clone)]
struct Foo(usize); //~ ERROR: struct `Foo` is never constructed
//^ NOTE: `Foo` is used in this macro
//| NOTE: `Foo` is used here

macro_rules! local_make_fn {
    ($name: ident) => {
        #[rustc_trivial_field_reads]
        fn local_foo() {
            println!("{}", $name(0).0);
        }
    }
}

local_make_fn!(Foo);

macro_def::make_fn!(Foo);

fn main() {
    local_foo();
    foo();
}
