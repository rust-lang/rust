#![feature(const_extern_fn)]

extern "C" {
    fn regular_in_block();
}

const extern fn bar() {
    unsafe {
        regular_in_block();
        //~^ ERROR: cannot call functions with `"C"` abi in `min_const_fn`
    }
}

extern fn regular() {}

const extern fn foo() {
    unsafe {
        regular();
        //~^ ERROR: cannot call functions with `"C"` abi in `min_const_fn`
    }
}

fn main() {}
