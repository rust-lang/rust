use std::fmt::{Debug, Display};

#[marker] trait ExplicitMarker {}
//~^ ERROR marker traits is an experimental feature (see issue #29864)

impl<T: Display> ExplicitMarker for T {}
impl<T: Debug> ExplicitMarker for T {}

fn main() {}
