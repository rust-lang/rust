//@ known-bug: rust-lang/rust #127304
#![feature(adt_const_params)]

trait Trait<T> {}
impl Trait<u16> for () {}

struct MyStr(str);
impl std::marker::ConstParamTy for MyStr {}

fn function_with_my_str<const S: &'static MyStr>() -> &'static MyStr {
    S
}

impl MyStr {
    const fn new(s: &Trait str) -> &'static MyStr {}
}

pub fn main() {
    let f = function_with_my_str::<{ MyStr::new("hello") }>();
}
