//@ known-bug: #140693
#![crate_type = "sdylib"]

pub trait Trait {
    type Type;
}
pub struct S;

#[export_stable]
pub extern "C" fn foo1(_x: <S as Trait>::Type) {}
