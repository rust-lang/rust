#![feature(associated_type_defaults)]

pub struct C<AType: A> {a:AType}

pub trait A {
    type B = C<Self::anything_here_kills_it>;
    //~^ ERROR: associated type `anything_here_kills_it` not found for `Self`
}

fn main() {}
