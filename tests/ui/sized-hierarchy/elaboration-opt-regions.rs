//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: old next
//@[next] compile-flags: -Znext-solver
#![feature(sized_hierarchy)]

use std::marker::{PhantomData, SizeOfVal, PointeeSized};

struct Foo<'a, T: PointeeSized>(PhantomData<&'a T>, T);

fn requires_sizeofval<T: SizeOfVal>() {}

fn foo<'a, T: 'a + PointeeSized>()
where
    Foo<'a, T>: Sized
{
    requires_sizeofval::<Foo<'_, T>>();
}
