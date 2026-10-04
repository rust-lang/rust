#![feature(freeze)]

//@ check-pass

use std::marker::Freeze;

trait Trait<T: Freeze + 'static> {
    const VALUE: T;
    const VALUE_REF: &'static T = &Self::VALUE;
}

fn main() {}
