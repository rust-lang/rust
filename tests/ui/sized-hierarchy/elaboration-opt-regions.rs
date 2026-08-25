//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: old next
//@[next] compile-flags: -Znext-solver
#![feature(sized_hierarchy)]

use std::marker::{PhantomData, MetaSized, PointeeSized};

struct Foo<'a, T: PointeeSized>(PhantomData<&'a T>, T);

fn requires_metasized<T: MetaSized>() {}

fn foo<'a, T: 'a + PointeeSized>()
where
    Foo<'a, T>: Sized
{
    requires_metasized::<Foo<'_, T>>();
}
