//@ check-pass
#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

trait Trait<const NAME: &'static str> {
    type Assoc;
}

impl Trait<"0"> for () {
    type Assoc = ();
}

fn main() {
    let _: <() as Trait<"0">>::Assoc = ();
}
