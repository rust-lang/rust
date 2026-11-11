mod foo {
    pub struct A;

    pub mod bar {
        pub struct B;
    }
}

use foo::{A, bar::B    as    A};
//~^ ERROR is defined multiple times

fn main() {}
