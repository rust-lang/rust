use std::fmt::{Debug, Display};

trait MyMarker {}

impl<T: Display> MyMarker for T {}
impl<T: Debug> MyMarker for T {}
//~^ ERROR E0119

fn main() {}
