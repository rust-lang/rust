#![feature(marker_trait_attr)]

#[marker]
trait Marker {}

impl Marker for &'_ () {} //~ ERROR type annotations needed
impl Marker for &'_ () {} //~ ERROR type annotations needed

fn main() {}
