//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: old next
//@[next] compile-flags: -Znext-solver
#![feature(sized_hierarchy)]

use std::marker::{PointeeSized, SizeOfVal};

trait Id: PointeeSized {
    type This: PointeeSized;
}

impl<T: PointeeSized> Id for T {
    type This = T;
}

fn requires_sizeofval<T: SizeOfVal>() {}

fn foo<T>()
where
    T: PointeeSized,
    <T as Id>::This: Sized
{
    // `T: Sized` from where bounds (`T: PointeeSized` removes any default bounds and
    // `<T as Id>::This: Sized` normalizes to `T: Sized`). This should trivially satisfy
    // `T: SizeOfVal`.
    requires_sizeofval::<T>();
}
