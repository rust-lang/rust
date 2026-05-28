// regression test for #127351

#![feature(lazy_type_alias)]

type ExplicitTypeOutlives<T> = T;

pub struct Warns {
    _significant_drop: ExplicitTypeOutlives,
    //~^ ERROR missing generics for type alias `ExplicitTypeOutlives`
    field: String,
}

pub fn test(w: Warns) {
    let _ = || drop(w.field);
}

fn main() {}
