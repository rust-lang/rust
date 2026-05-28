//@ check-pass
//@ compile-flags: -Csymbol-mangling-version=v0
#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

// Regression test for #116303

#[derive(PartialEq, Eq)]
struct MyStr(str);
impl std::marker::ConstParamTy_ for MyStr {}

fn function_with_my_str<const S: &'static MyStr>() -> &'static MyStr {
    S
}

impl MyStr {
    const fn new(s: &'static str) -> &'static MyStr {
        unsafe { std::mem::transmute(s) }
    }
}

pub fn main() {
    let f = function_with_my_str::<{ MyStr::new("hello") }>();
}
