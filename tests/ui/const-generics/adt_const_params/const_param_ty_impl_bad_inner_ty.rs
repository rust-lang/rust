#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

// #112124

struct Foo;

impl ConstParamTy for &Foo {}
//~^ ERROR the trait `ConstParamTy` cannot be implemented for this type

impl ConstParamTy for &[Foo] {}
//~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
//~| ERROR the trait `ConstParamTy` cannot be implemented for this type

impl ConstParamTy for [Foo; 4] {}
//~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
//~| ERROR the trait `ConstParamTy` cannot be implemented for this type

impl ConstParamTy for (Foo, i32, *const u8) {}
//~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
//~| ERROR the trait `ConstParamTy` cannot be implemented for this type

// #119299 (ICE)

#[derive(Eq, PartialEq)]
struct Wrapper(*const i32, usize);

impl ConstParamTy for &Wrapper {}
//~^ ERROR the trait `ConstParamTy` cannot be implemented for this type

const fn foo<const S: &'static Wrapper>() {}

fn main() {
    const FOO: Wrapper = Wrapper(&42 as *const i32, 42);
    foo::<{ &FOO }>();
}
