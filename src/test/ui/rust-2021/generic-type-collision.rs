// check-pass
// run-rustfix
// edition 2018
#![warn(future_prelude_collision)]

trait MyTrait<A> {
    fn from_iter(x: Option<A>);
}

impl<T> MyTrait<()> for Vec<T> {
    fn from_iter(_: Option<()>) {}
}

fn main() {
    <Vec<i32>>::from_iter(None);
    //~^ WARNING trait-associated function `from_iter` will become ambiguous in Rust 2021
    //~^^ WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in the 2021 edition!
}
