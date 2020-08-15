// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn main() {
    let _: UnknownStruct<7>;
    //~^ ERROR cannot find type `UnknownStruct`
}
