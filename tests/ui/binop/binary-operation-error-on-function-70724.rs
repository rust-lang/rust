// https://github.com/rust-lang/rust/issues/70724
fn a() -> i32 {
    3
}

pub fn main() {
    assert_eq!(a, 0);
    //~^ ERROR binary operation `==` cannot
    //~| ERROR mismatched types
    //~| ERROR doesn't implement
}
