#![feature(const_fn)]

const fn foo() { (||{})() }
//~^ ERROR calls in constant functions are limited to constant functions, tuple structs and tuple
// variants

const fn bad(input: fn()) {
    input()
    //~^ ERROR function pointers are not allowed in const fn
}

fn main() {
}
