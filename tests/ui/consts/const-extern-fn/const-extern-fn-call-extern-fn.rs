extern "C" {
    fn regular_in_block();
}

const extern "C" fn bar() {
    unsafe {
        regular_in_block();
        //~^ ERROR: cannot call non-const function
    }
}

extern "C" fn regular() {}

const extern "C" fn foo() {
    unsafe {
        regular();
        //~^ ERROR: cannot call non-const function
    }
}

fn main() {}
