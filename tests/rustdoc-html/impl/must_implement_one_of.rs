#![crate_name = "c"]
#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(a, b)]
//@ matches c/trait.Trait.html '//*[@class="stab must_implement"]' \
//      'At least one of the `a`, `b` methods is required.$'
pub trait Trait {
    fn a() {}
    fn b() {}
}
