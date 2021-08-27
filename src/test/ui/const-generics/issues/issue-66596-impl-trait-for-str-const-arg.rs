// check-pass
#![feature(const_param_types)]
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
