// https://github.com/rust-lang/rust/issues/50585
fn main() {
    |y: Vec<[(); for x in 0..2 {}]>| {};
    //~^ ERROR mismatched types
}
