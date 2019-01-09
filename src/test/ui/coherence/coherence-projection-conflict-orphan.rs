// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(rustc_attrs)]

// Here we expect a coherence conflict because, even though `i32` does
// not implement `Iterator`, we cannot rely on that negative reasoning
// due to the orphan rules. Therefore, `A::Item` may yet turn out to
// be `i32`.

pub trait Foo<P> { fn foo() {} }

pub trait Bar {
    type Output: 'static;
}

impl Foo<i32> for i32 { }

impl<A:Iterator> Foo<A::Item> for A { }
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119

fn main() {}
