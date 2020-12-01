#![feature(const_fn_fn_ptr_basics)]

const fn foo() { (||{})() }
//~^ ERROR calls in constant functions

const fn bad(input: fn()) {
    input()
    //~^ ERROR function pointer
}

fn main() {
}
