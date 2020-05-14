// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait Trait<const NAME: &'static str> {
    type Assoc;
}

impl Trait<"0"> for () {
    type Assoc = ();
}

fn main() {
    let _: <() as Trait<"0">>::Assoc = ();
}
