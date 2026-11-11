// test for https://github.com/rust-lang/rust/issues/86940
//@ run-rustfix
//@ edition:2018
//@ check-pass
#![warn(rust_2021_prelude_collisions)]
#![allow(dead_code)]
#![allow(unused_imports)]

struct Generic<'a, U>(&'a U);

trait MyFromIter {
    fn from_iter(_: i32) -> Self;
}

impl MyFromIter for Generic<'static, i32> {
    fn from_iter(_: i32) -> Self {
        todo!()
    }
}

impl std::iter::FromIterator<i32> for Generic<'static, i32> {
    fn from_iter<T: IntoIterator<Item = i32>>(_: T) -> Self {
        todo!()
    }
}

fn main() {
    Generic::from_iter(1);
    //~^ WARNING trait-associated function `from_iter` will become ambiguous in Rust 2021
    //~| WARN this is accepted in the current edition (Rust 2018)
    Generic::<'static, i32>::from_iter(1);
    //~^ WARNING trait-associated function `from_iter` will become ambiguous in Rust 2021
    //~| WARN this is accepted in the current edition (Rust 2018)
    Generic::<'_, _>::from_iter(1);
    //~^ WARNING trait-associated function `from_iter` will become ambiguous in Rust 2021
    //~| WARN this is accepted in the current edition (Rust 2018)
}
