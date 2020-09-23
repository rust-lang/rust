#![feature(const_extern_fn)]

extern "C" {
    fn regular_in_block();
}

const extern fn bar() {
    unsafe {
        regular_in_block();
        //~^ ERROR: calls in constant functions
    }
}

extern fn regular() {}

const extern fn foo() {
    unsafe {
        regular();
        //~^ ERROR: calls in constant functions
    }
}

fn main() {}
