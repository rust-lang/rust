//@ compile-flags: -Znext-solver
//@ check-pass
//@ known-bug: #130516
// An unsafe binder is `Sized` if and only if its inner type is `Sized`.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::mem::ManuallyDrop;

trait Tr {
    type Assoc<'a>: ?Sized;
}

fn requires_sized<T: Sized>() {}

fn sized_bound() {
    requires_sized::<unsafe<> ManuallyDrop<[u8]>>();
    requires_sized::<unsafe<'a> ManuallyDrop<dyn Debug + 'a>>();
}

// The inner type is only known to be `Sized` from the where-clauses, so these
// go through the solvers' structural `Sized` impls instead of the fast path.
fn sized_by_where_clause<T: Tr, U: ?Sized>()
where
    for<'a> T::Assoc<'a>: Sized,
    U: Sized,
{
    requires_sized::<unsafe<'a> ManuallyDrop<T::Assoc<'a>>>();
    requires_sized::<unsafe<'a> ManuallyDrop<(&'a u8, U)>>();
}

fn not_sized_generic<T: Tr, U: ?Sized>() {
    requires_sized::<unsafe<'a> ManuallyDrop<T::Assoc<'a>>>();
    requires_sized::<unsafe<'a> ManuallyDrop<(&'a u8, U)>>();
}

fn by_value(x: unsafe<> ManuallyDrop<[u8]>) -> unsafe<> ManuallyDrop<[u8]> {
    x
}

fn move_out_of_box(x: Box<unsafe<> ManuallyDrop<[u8]>>) {
    let _y = *x;
}

fn size<T>() -> usize {
    std::mem::size_of::<T>()
}

fn main() {
    size::<unsafe<> ManuallyDrop<[u8]>>();
}
