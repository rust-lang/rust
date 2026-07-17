//@ known-bug: unknown
//@revisions: stable gated
#![cfg_attr(gated, feature(const_trait_impl))]

const fn cmp(x: fn(&'static ()), y: for<'a> fn(&'a ())) -> bool {
    x == y
}

fn main() {}
