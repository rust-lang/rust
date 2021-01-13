// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn main() {
    let _: UnknownStruct<7>;
    //~^ ERROR cannot find type `UnknownStruct`
}
