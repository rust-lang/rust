//@ check-pass

#![feature(negative_bounds)]

trait A: !B {}
trait B: !A {}

fn main() {}
