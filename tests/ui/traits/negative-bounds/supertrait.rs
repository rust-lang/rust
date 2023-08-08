// check-pass

#![feature(negative_bounds)]
//~^ WARN the feature `negative_bounds` is incomplete

trait A: !B {}
trait B: !A {}

fn main() {}
