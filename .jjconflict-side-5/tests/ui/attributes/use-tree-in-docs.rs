// Don't ICE when use tree appears in docs. (#163355)
#![feature(generic_assert)]
#[doc = assert !(b)]
//~^ ERROR attribute value must be a literal
use std as x;

fn main(){}
