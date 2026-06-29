//! Test that using `#[splat]` on un-resolvable types is an error.

#![allow(incomplete_features)]
#![allow(unconditional_recursion)]
#![feature(splat)]
#![feature(tuple_trait)]

fn tuple(#[splat] t: impl Sized) -> impl Sized {
    //~^ ERROR cannot resolve opaque type
    tuple(tuple((t, ())))
}

fn tuple_trait(#[splat] t: impl std::marker::Tuple) -> impl std::marker::Tuple {
    //~^ ERROR cannot resolve opaque type
    tuple_trait(tuple_trait((t, ())))
}

trait Trait {
    type MaybeTup;
    type Tup: std::marker::Tuple;
}

fn ambig(#[splat] t: Trait::MaybeTup) {}
//~^ ERROR ambiguous associated type
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
fn ambig_tup(#[splat] t: Trait::Tup) {}
//~^ ERROR ambiguous associated type
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a

fn main() {
    tuple();
    tuple_trait();
    ambig();
    ambig_tup();

    tuple(1);
    tuple_trait(1);
    ambig(1);
    ambig_tup(1);

    tuple(1, 2.0);
    tuple_trait(1, 2.0);
    ambig(1, 2.0);
    ambig_tup(1, 2.0);
}
