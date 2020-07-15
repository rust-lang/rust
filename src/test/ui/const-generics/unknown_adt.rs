#![feature(const_generics)]
#![allow(incomplete_features)]

fn main() {
    let _: UnknownStruct<7>;
    //~^ ERROR cannot find type `UnknownStruct`
}
