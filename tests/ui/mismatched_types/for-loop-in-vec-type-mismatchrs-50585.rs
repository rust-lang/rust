// https://github.com/rust-lang/rust/issues/50585
fn main() {
    |y: Vec<[(); for x in 0..2 {}]>| {};
    //~^ ERROR mismatched types
    //~| ERROR  `std::ops::Range<{integer}>: const Iterator` is not satisfied
    //~| ERROR  `std::ops::Range<{integer}>: const Iterator` is not satisfied
}
