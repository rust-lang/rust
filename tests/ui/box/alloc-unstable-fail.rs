use std::boxed::Box;

fn main() {
    let _boxed: Box<u32, _> = Box::new(10);
    //~^ ERROR use of unstable library feature 'allocator_api'
}
