use std::mem;

trait Misc {}

fn size_of_copy<T: Copy+?Sized>() -> usize { mem::size_of::<T>() }

fn main() {
    size_of_copy::<dyn Misc + Copy>();
    //~^ ERROR only auto traits can be used as additional traits in a trait object
    //~| ERROR the trait bound `dyn Misc: std::marker::Copy` is not satisfied
}
