#![feature(type_info)]

use std::any::TypeId;

fn main() {}

const _: () = const {
    TypeId::of::<Option::<()>>().fields(2);
    //~^ ERROR indexing out of bounds: the len is 2 but the index is 2
};

const _: () = const {
    TypeId::of::<Option::<()>>().field(0, 1);
    //~^ ERROR indexing out of bounds: the len is 0 but the index is 1
};
