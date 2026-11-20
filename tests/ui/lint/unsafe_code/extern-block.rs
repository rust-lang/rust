//@ edition:2021
#![deny(unsafe_code)]

#[allow(unsafe_code)]
extern "C" {
    fn foo();
}

extern "C" {
    //~^ ERROR usage of an `extern` block [unsafe_code]
    fn bar();
}

fn main() {}
