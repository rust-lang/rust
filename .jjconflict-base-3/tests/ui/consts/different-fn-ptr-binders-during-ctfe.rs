//@revisions: stable gated
#![cfg_attr(gated, feature(const_trait_impl))]

const fn cmp(x: fn(&'static ()), y: for<'a> fn(&'a ())) -> bool {
    x == y
    //~^ ERROR pointers cannot be reliably compared during const eval
}

fn main() {}
