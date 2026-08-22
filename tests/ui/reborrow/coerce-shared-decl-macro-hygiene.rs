//! Test that Reborrow and CoerceShared can be derived in macros.
//! This should eventually pass.


#![feature(reborrow, decl_macro)]
#![allow(incomplete_features)]

use std::marker::{CoerceShared, Reborrow};

macro my_macro($field:ident) {
    pub struct MyMut<'a> {
        $field: &'a i32,
        field: &'a i64,
    }

    #[derive(Clone, Copy)]
    pub struct MyRef<'a> {
        $field: &'a i32,
        field: &'a i64,
    }

    impl Reborrow for MyMut<'_> {}

    impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}
    //~^ ERROR
}

my_macro!(field);

fn main() {}
