#![feature(box_into_boxed_slice)]

use std::boxed::Box;
use std::fmt::Debug;
fn main() {
    let boxed_slice = Box::new([1,2,3]) as Box<[u8]>;
    let _ = Box::into_boxed_slice(boxed_slice);
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    //~^^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    let boxed_trait: Box<dyn Debug> = Box::new(5u8);
    let _ = Box::into_boxed_slice(boxed_trait);
    //~^ ERROR the size for values of type `dyn Debug` cannot be known at compilation time
    //~^^ ERROR the size for values of type `dyn Debug` cannot be known at compilation time
}
