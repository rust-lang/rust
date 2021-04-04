//[full] check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]


trait Trait<const NAME: &'static str> {
//[min]~^ ERROR `&'static str` is forbidden
    type Assoc;
}

impl Trait<"0"> for () {
    type Assoc = ();
}

fn main() {
    let _: <() as Trait<"0">>::Assoc = ();
}
