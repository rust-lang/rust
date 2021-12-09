#![feature(const_extern_fn)]

extern "C" {
    fn regular_in_block();
}

const extern "C" fn bar() {
    unsafe {
        regular_in_block();
        //~^ ERROR: cannot call non-const fn
    }
}

extern "C" fn regular() {}

const extern "C" fn foo() {
    unsafe {
        regular();
        //~^ ERROR: cannot call non-const fn
    }
}

fn main() {}
