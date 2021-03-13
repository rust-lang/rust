// aux-build:send_sync.rs

#![feature(trait_alias)]

extern crate send_sync;

use std::rc::Rc;
use send_sync::SendSync;

fn use_alias<T: SendSync>() {}

fn main() {
    use_alias::<u32>();
    use_alias::<Rc<u32>>();
    //~^ ERROR `Rc<u32>` cannot be sent between threads safely [E0277]
    //~^^ ERROR `Rc<u32>` cannot be shared between threads safely [E0277]
}
