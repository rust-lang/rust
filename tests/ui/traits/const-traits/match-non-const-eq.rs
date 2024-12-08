//@[gated] check-pass
//@ revisions: gated stock

#![cfg_attr(gated, feature(const_trait_impl))]

const fn foo(input: &'static str) {
    match input {
        "a" => (),
        //[stock]~^ cannot call conditionally-const method `<str as PartialEq>::eq` in constant functions
        _ => (),
    }
}

fn main() {}
