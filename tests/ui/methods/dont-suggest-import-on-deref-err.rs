use std::clone::Clone;
use std::ops::Deref;

#[derive(Clone)]
pub struct Foo {}

impl Deref for Foo {}
//~^ ERROR not all trait items implemented

pub fn main() {
    let f = Foo {};
    let _ = f.clone();
}
