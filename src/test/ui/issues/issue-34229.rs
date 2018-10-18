#[derive(PartialEq)] struct Comparable;
#[derive(PartialEq, PartialOrd)] struct Nope(Comparable);

fn main() {}
