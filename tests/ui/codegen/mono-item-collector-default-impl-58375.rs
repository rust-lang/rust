// https://github.com/rust-lang/rust/issues/58375
// Make sure that the mono-item collector does not crash when trying to
// instantiate a default impl for DecodeUtf16<<u8 as A>::Item>
// See https://github.com/rust-lang/rust/issues/58375

//@ build-pass
//@ compile-flags:-C link-dead-code

#![crate_type = "rlib"]

pub struct DecodeUtf16<I>(I);

pub trait Arbitrary {
    fn arbitrary() {}
}

pub trait A {
    type Item;
}

impl A for u8 {
    type Item = char;
}

impl Arbitrary for DecodeUtf16<<u8 as A>::Item> {}
