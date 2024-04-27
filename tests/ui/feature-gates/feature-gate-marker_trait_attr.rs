use std::fmt::{Debug, Display};

#[marker] trait ExplicitMarker {}
//~^ ERROR the `#[marker]` attribute is an experimental feature

impl<T: Display> ExplicitMarker for T {}
impl<T: Debug> ExplicitMarker for T {}

fn main() {}
