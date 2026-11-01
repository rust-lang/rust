//@ check-pass
//@ revisions: stock cnst
#![cfg_attr(cnst, feature(const_trait_impl))]

const _: fn(&String) = |s| {
    &*s as &str;
};

fn main() {}
