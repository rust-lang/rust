use std::mem;

trait Misc {}

fn size_of_copy<T: Copy + ?Sized>() -> usize { mem::size_of::<T>() }

fn main() {
    size_of_copy::<dyn Misc + Copy>();
    //~^ ERROR only auto traits can be used as additional traits in a trait object [E0225]
    //~| ERROR the trait bound `dyn std::marker::Copy: std::marker::Copy` is not satisfied [E0277]
    //~| ERROR the trait `std::marker::Copy` cannot be made into an object [E0038]
}
