fn main() {
    let u: usize = std::mem:size_of::<u32>();
    //~^ ERROR casts cannot be followed by a function call
    //~| ERROR expected value, found module `std::mem` [E0423]
    //~| ERROR cannot find type `size_of` in this scope [E0412]
}
