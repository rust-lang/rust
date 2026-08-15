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

impl<'a> CoerceShared<ForeignPtrRef<'a>> for LocalPtrMut<'a> {}
//~^ ERROR

fn main() {}
