#![feature(const_extern_fn)]

extern "C" {
    fn regular_in_block();
}

const extern fn bar() {
    unsafe {
        regular_in_block();
        //~^ ERROR: can only call other `const fn` within a `const fn`
    }
}

extern fn regular() {}

const extern fn foo() {
    unsafe {
        regular();
        //~^ ERROR: can only call other `const fn` within a `const fn`
    }
}

fn main() {}
