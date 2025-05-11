//@ check-pass

#![deny(clippy::multiple_inherent_impl)]

// Test for https://github.com/rust-lang/rust-clippy/issues/4578

macro_rules! impl_foo {
    ($struct:ident) => {
        impl $struct {
            fn foo() {}
        }
    };
}

macro_rules! impl_bar {
    ($struct:ident) => {
        impl $struct {
            fn bar() {}
        }
    };
}

struct MyStruct;

impl_foo!(MyStruct);
impl_bar!(MyStruct);

fn main() {}
