//! Regression test for https://github.com/rust-lang/rust/issues/11047

//@ run-pass
// Test that static methods can be invoked on `type` aliases

#![allow(unused_variables)]

pub mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Qux;

            impl Qux {
                pub fn new() {}
            }
        }
    }
}

fn main() {

    type Ham = foo::bar::baz::Qux;
    let foo: () = foo::bar::baz::Qux::new();  // invoke directly
    let bar: () = Ham::new();                 // invoke via type alias

    type StringVec = Vec<String>;
    let sv = StringVec::new();
}
