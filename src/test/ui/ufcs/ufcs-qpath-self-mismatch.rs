use std::ops::Add;

fn main() {
    <i32 as Add<u32>>::add(1, 2);
    //~^ ERROR cannot add `u32` to `i32`
    <i32 as Add<i32>>::add(1u32, 2);
    //~^ ERROR arguments to this function are incorrect
    <i32 as Add<i32>>::add(1, 2u32);
    //~^ ERROR arguments to this function are incorrect
}
