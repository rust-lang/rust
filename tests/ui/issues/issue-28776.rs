use std::ptr;

fn main() {
    (&ptr::write)(1 as *mut _, 42);
    //~^ ERROR E0133
}
