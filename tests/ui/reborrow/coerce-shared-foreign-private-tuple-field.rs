//@ normalize-stderr: "\n\n\z" -> "\n"

//@ aux-build: reborrow_foreign_private.rs

#![feature(reborrow)]

extern crate reborrow_foreign_private;

use reborrow_foreign_private::ForeignPtrRef;
use std::marker::{CoerceShared, PhantomData, Reborrow};
use std::ptr;

struct LocalPtrMut<'a>((*const i32, PhantomData<&'a ()>));

impl<'a> Reborrow for LocalPtrMut<'a> {}

impl<'a> CoerceShared<ForeignPtrRef<'a>> for LocalPtrMut<'a> {}
//~^ ERROR

fn main() {
    let local = LocalPtrMut((ptr::null(), PhantomData));
    let foreign: ForeignPtrRef<'_> = local;
    let _ = foreign.to_ref();
}
