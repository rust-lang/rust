mod foo {
    pub struct A;
    pub struct B;
}

use foo::{A, A};
//~^ ERROR is defined multiple times

fn main() {}
