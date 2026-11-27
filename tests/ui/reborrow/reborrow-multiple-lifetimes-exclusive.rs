//! A `Reborrow` type with two lifetimes and no reborrowed field is borrowed as a
//! whole, under both lifetimes, so two live reborrows still conflict.

#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

struct Marker<'a, 'b>(PhantomData<(&'a mut (), &'b mut ())>);

impl<'a, 'b> Reborrow for Marker<'a, 'b> {}

fn reborrow<'a, 'b>(m: Marker<'a, 'b>) -> Marker<'a, 'b> {
    m
}

fn keep(_: &Marker<'_, '_>) {}

fn main() {
    let m = Marker(PhantomData);
    let x = reborrow(m);
    let y = reborrow(m); //~ ERROR cannot borrow `m` as mutable more than once at a time
    keep(&x);
    keep(&y);
}
