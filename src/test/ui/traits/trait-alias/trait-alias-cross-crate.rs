// aux-build:trait_alias.rs

#![feature(trait_alias)]

extern crate trait_alias;

use std::rc::Rc;
use trait_alias::SendSync;

fn use_alias<T: SendSync>() {}

fn main() {
    use_alias::<u32>();
    use_alias::<Rc<u32>>();
    //~^ ERROR `std::rc::Rc<u32>` cannot be sent between threads safely [E0277]
    //~^^ ERROR `std::rc::Rc<u32>` cannot be shared between threads safely [E0277]
}
