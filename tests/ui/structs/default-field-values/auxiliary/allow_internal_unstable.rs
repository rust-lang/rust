#![allow(internal_features)]
#![feature(allow_internal_unstable)]

#[macro_export]
#[allow_internal_unstable(default_field_values)]
macro_rules! foo {
    () => { fn foo() { struct Foo {x: usize = 0, } Foo {..}; } };
}
