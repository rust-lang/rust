#[derive(PartialEq)] struct Comparable;
#[derive(PartialEq, PartialOrd)] struct Nope(Comparable);
//~^ ERROR can't compare `Comparable`

fn main() {}

// https://github.com/rust-lang/rust/issues/34229
