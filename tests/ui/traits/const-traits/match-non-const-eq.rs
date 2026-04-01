//@ revisions: stock gated
#![cfg_attr(gated, feature(const_trait_impl, const_cmp))]
//@[gated] check-pass

const fn foo(input: &'static str) {
    match input {
        "a" => (),
        //[stock]~^ ERROR cannot match on `str` in constant functions
        //[stock]~| ERROR `PartialEq` is not yet stable as a const trait
        _ => (),
    }
}

fn main() {}
