// revisions: stock gated
#![cfg_attr(gated, feature(const_trait_impl))]

const fn foo(input: &'static str) {
    match input {
        "a" => (), //[gated]~ ERROR can't compare `str` with `str` in const contexts
        //~^ ERROR cannot match on `str` in constant functions
        _ => (),
    }
}

fn main() {}
