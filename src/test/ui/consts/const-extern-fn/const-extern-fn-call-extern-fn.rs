#![feature(const_extern_fn)]

extern "C" {
    fn regular_in_block();
}

const extern "C" fn bar() {
    unsafe {
        regular_in_block();
        //~^ ERROR: calls in constant functions
    }
}

extern "C" fn regular() {}

const extern "C" fn foo() {
    unsafe {
        regular();
        //~^ ERROR: calls in constant functions
    }
}

fn main() {}
