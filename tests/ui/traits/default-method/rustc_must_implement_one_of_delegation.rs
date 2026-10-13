//@ check-pass

#![feature(rustc_attrs)]
#![feature(fn_delegation)]

mod to_reuse {
    pub fn b() {}
    pub fn c() {}
}

#[rustc_must_implement_one_of(a, b)]
trait TraitDelegation {
    fn a() {}
    reuse to_reuse::b;
}

#[rustc_must_implement_one_of(a, ren)]
trait TraitDelegationRenamed {
    fn a() {}
    reuse to_reuse::c as ren;
}

fn main() {}
