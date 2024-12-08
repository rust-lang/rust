#[derive(PartialEq)] struct Comparable;
#[derive(PartialEq, PartialOrd)] struct Nope(Comparable);
//~^ ERROR can't compare `Comparable`

fn main() {}
