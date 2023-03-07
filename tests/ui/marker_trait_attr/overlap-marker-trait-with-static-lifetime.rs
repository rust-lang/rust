// check-pass
#![feature(marker_trait_attr)]

#[marker]
trait Marker {}

impl Marker for &'static () {}
impl Marker for &'static () {}

fn main() {}
