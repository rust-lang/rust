//@ check-pass
//! Types with more than one lifetime where borrowck finds no field to reborrow.
//! Borrowck used to hit a `bug!` on all of these. Where a value is used twice, the
//! second use only compiles if the first one was a reborrow and not a move.

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

// Only a marker field.
struct Marker<'a, 'b>(PhantomData<(&'a mut (), &'b mut ())>);

impl<'a, 'b> Reborrow for Marker<'a, 'b> {}

fn marker<'a, 'b>(m: Marker<'a, 'b>) -> Marker<'a, 'b> {
    m
}

// A type parameter field.
struct Generic<'a, 'b, T> {
    t: T,
    marker: PhantomData<(&'a (), &'b ())>,
}

impl<'a, 'b, T: Reborrow> Reborrow for Generic<'a, 'b, T> {}

fn generic<'a, 'b>(g: Generic<'a, 'b, Inner<'a>>) -> Generic<'a, 'b, Inner<'a>> {
    g
}

// An associated type field that normalizes to `&mut`.
trait HasAssoc<'a> {
    type Assoc;
}

impl<'a> HasAssoc<'a> for () {
    type Assoc = &'a mut ();
}

struct Assoc<'a, 'b> {
    assoc: <() as HasAssoc<'a>>::Assoc,
    shared: &'b (),
}

impl<'a, 'b> Reborrow for Assoc<'a, 'b> {}

fn assoc<'a, 'b>(x: Assoc<'a, 'b>) -> Assoc<'a, 'b> {
    x
}

// A field that is only `Copy` when `T: Copy`, used in a body with that bound.
#[derive(Clone, Copy)]
struct Wrap<'a, T>(PhantomData<&'a ()>, T);

impl<'a, T: Reborrow> Reborrow for Wrap<'a, T> {}

struct WithWrap<'a, 'b, T> {
    w: Wrap<'a, T>,
    shared: &'b (),
}

impl<'a, 'b, T: Reborrow> Reborrow for WithWrap<'a, 'b, T> {}

fn with_wrap<'a, 'b, T: Reborrow + Copy>(x: WithWrap<'a, 'b, T>) -> WithWrap<'a, 'b, T> {
    x
}

// A `CoerceShared` target with more lifetimes than the source.
struct Src<'a>(PhantomData<(&'a (), &'a ())>);

impl<'a> Reborrow for Src<'a> {}

#[derive(Clone, Copy)]
struct Dst<'a, 'b>(PhantomData<(&'a (), &'b ())>);

impl<'a> CoerceShared<Dst<'a, 'a>> for Src<'a> {}

fn coerce<'a>(x: Src<'a>) -> Dst<'a, 'a> {
    x
}

struct Inner<'a>(&'a mut ());

impl<'a> Reborrow for Inner<'a> {}

fn main() {
    let m = Marker(PhantomData);
    let _ = marker(m);
    let _ = marker(m);

    let mut a = ();
    let g = Generic { t: Inner(&mut a), marker: PhantomData };
    let _ = generic(g);
    let _ = generic(g);

}
