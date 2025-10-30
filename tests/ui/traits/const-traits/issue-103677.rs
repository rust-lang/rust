//@[stock] check-pass
//@ revisions: stock cnst
#![cfg_attr(cnst, feature(const_trait_impl))]

const _: fn(&String) = |s| {
    &*s as &str;
    //[cnst]~^ ERROR: the trait bound `String: const Deref` is not satisfied
};

fn main() {}
