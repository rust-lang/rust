//@ check-pass

//! Test that CoerceShared cannot be implemented targeting a foreign tuple struct with private
//! fields.

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

mod foreign_ptr {
    use std::marker::PhantomData;

    #[derive(Clone, Copy)]
    pub struct ForeignPtrRef<'a>(*const i32, PhantomData<&'a ()>);
}

use foreign_ptr::ForeignPtrRef;

struct LocalPtrMut<'a>(*const i32, PhantomData<&'a ()>);

impl<'a> Reborrow for LocalPtrMut<'a> {}

// Should error: ForeignPtrRef has private fields.
impl<'a> CoerceShared<ForeignPtrRef<'a>> for LocalPtrMut<'a> {}

fn main() {}
