//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::ops::Deref;

trait Trait {}
impl<A, B> Trait for (A, B, u8) where A: Deref, B: Deref<Target = A::Target>, {}
impl<A, B> Trait for (A, B, i8) {}

type TaitSized = impl Sized;
#[define_opaque(TaitSized)]
fn def_tait1() -> TaitSized {}

type TaitCopy = impl Copy;
#[define_opaque(TaitCopy)]
fn def_tait2() -> TaitCopy {}

fn impl_trait<T: Trait> () {}

fn test() {
    impl_trait::<(&TaitSized, &TaitCopy, _)>();
    impl_trait::<(&TaitCopy, &TaitSized, _)>();

    impl_trait::<(&TaitCopy, &String, _)>();
    impl_trait::<(&TaitSized, &String, _)>();
}

fn main() {}
