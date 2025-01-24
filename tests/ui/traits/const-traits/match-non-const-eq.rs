//@ known-bug: #110395
//@ revisions: stock gated
#![cfg_attr(gated, feature(const_trait_impl))]

const fn foo(input: &'static str) {
    match input {
        "a" => (), //FIXME [gated]~ ERROR can't compare `str` with `str` in const contexts
        //FIXME ~^ ERROR cannot match on `str` in constant functions
        _ => (),
    }
}

fn main() {}
