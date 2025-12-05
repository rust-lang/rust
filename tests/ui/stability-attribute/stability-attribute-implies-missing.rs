#![feature(staged_api)]
#![stable(feature = "stability_attribute_implies", since = "1.0.0")]

// Tests that `implied_by = "bar"` results in an error being emitted if `bar` does not exist.

#[unstable(feature = "foobar", issue = "1", implied_by = "bar")]
//~^ ERROR feature `bar` implying `foobar` does not exist
pub fn foobar() {}

fn main() {}
