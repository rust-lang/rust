//@ run-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{PhantomData, Reborrow};

#[derive(Reborrow)]
struct Named<'a, T> {
    value: &'a mut T,
}

#[derive(Reborrow)]
struct Tuple<'a, T>(&'a mut T);

#[derive(Reborrow)]
struct Generic<'a, T, const N: usize>
where
    T: 'a,
{
    value: &'a mut [T; N],
}

#[derive(Reborrow)]
struct Marker<'a>(PhantomData<&'a ()>);

fn take_named(_: Named<'_, ()>) {}
fn take_tuple(_: Tuple<'_, ()>) {}
fn take_generic(_: Generic<'_, (), 1>) {}
fn take_marker<'a>(_: Marker<'a>) -> &'a () {
    &()
}

fn main() {
    let named = Named { value: &mut () };
    take_named(named);
    take_named(named);

    let tuple = Tuple(&mut ());
    take_tuple(tuple);
    take_tuple(tuple);

    let generic = Generic { value: &mut [()] };
    take_generic(generic);
    take_generic(generic);

    let marker = Marker(PhantomData);
    let _ = take_marker(marker);
    let _ = take_marker(marker);
}
