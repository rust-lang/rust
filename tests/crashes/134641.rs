//@ known-bug: #134641
#![feature(associated_const_equality)]

pub trait IsVoid {
    const IS_VOID: bool;
}
impl IsVoid for () {
    const IS_VOID: bool = true;
}

pub trait Maybe {}
impl Maybe for () {}
impl Maybe for () where (): IsVoid<IS_VOID = true> {}
