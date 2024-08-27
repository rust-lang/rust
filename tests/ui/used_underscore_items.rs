//@aux-build:external_item.rs
#![allow(unused)]
#![warn(clippy::used_underscore_items)]

extern crate external_item;

// should not lint macro
macro_rules! macro_wrap_func {
    () => {
        fn _marco_foo() {}
    };
}

macro_wrap_func!();

struct _FooStruct {}

impl _FooStruct {
    fn _method_call(self) {}
}

fn _foo1() {}

fn _foo2() -> i32 {
    0
}

mod a {
    pub mod b {
        pub mod c {
            pub fn _foo3() {}

            pub struct _FooStruct2 {}

            impl _FooStruct2 {
                pub fn _method_call(self) {}
            }
        }
    }
}

fn main() {
    _foo1();
    let _ = _foo2();
    a::b::c::_foo3();
    let _ = &_FooStruct {};
    let _ = _FooStruct {};

    let foo_struct = _FooStruct {};
    foo_struct._method_call();

    let foo_struct2 = a::b::c::_FooStruct2 {};
    foo_struct2._method_call();
}

// should not lint exteranl crate.
// user cannot control how others name their items
fn external_item_call() {
    let foo_struct3 = external_item::_ExternalStruct {};
    foo_struct3._foo();

    external_item::_exernal_foo();
}
