#![deny(unsafe_code)]

#[allow(unsafe_code)]
unsafe extern "C" {
    fn foo();
}

unsafe extern "C" {
    //~^ ERROR usage of an `unsafe extern` block [unsafe_code]
    fn bar();
}

fn main() {}
