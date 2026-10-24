#![feature(rustc_attrs, unboxed_closures)]
#![allow(internal_features)]

#[rustc_paren_sugar]
trait Tr {
    fn method();
}

fn main() {
    <u8 as Tr>::method();
    //~^ ERROR the trait bound `u8: Tr` is not satisfied
}
