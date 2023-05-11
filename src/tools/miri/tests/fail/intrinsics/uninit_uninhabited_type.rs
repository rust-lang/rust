#![feature(never_type)]

#[allow(deprecated, invalid_value)]
fn main() {
    unsafe { std::mem::uninitialized::<!>() };
    //~^ ERROR: attempted to instantiate uninhabited type `!`
}
