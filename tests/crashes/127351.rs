//@ known-bug: #127351
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

struct Outer0<'a, T>(ExplicitTypeOutlives<'a, T>);
type ExplicitTypeOutlives<'a, T: 'a> = (&'a (), T);

pub struct Warns {
    _significant_drop: ExplicitTypeOutlives,
    field: String,
}

pub fn test(w: Warns) {
    _ = || drop(w.field);
}

fn main() {}
