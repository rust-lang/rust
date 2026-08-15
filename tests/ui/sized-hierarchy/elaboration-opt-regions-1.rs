//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: old next
//@[next] compile-flags: -Znext-solver
#![feature(sized_hierarchy)]

use std::marker::{PhantomData, MetaSized, PointeeSized};

struct Foo<'a, T: PointeeSized>(*mut &'a (), T);

fn requires_metasized<'a, T: MetaSized>(f: &'a T) {}

fn foo<'a, T: PointeeSized>(f: &Foo<'a, T>)
where
    Foo<'a, T>: Sized
{
    requires_metasized(f);
}
