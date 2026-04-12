//@ build-pass

#![feature(unsafe_binders)]

fn panic<T>() { panic!() }

fn main() { panic::<unsafe<'a> &'a ()>(); }
