#![crate_type="lib"]
#![feature(unboxed_closures)]

pub trait Foo {
    extern "rust-call" fn foo(&self, _: ()) -> i32;
    extern "rust-call" fn foo_(&self, _: ()) -> i32 { 0 }
}
