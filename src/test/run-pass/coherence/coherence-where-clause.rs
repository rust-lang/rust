// run-pass
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

use std::fmt::Debug;
use std::default::Default;

trait MyTrait {
    fn get(&self) -> Self;
}

impl<T> MyTrait for T
    where T : Default
{
    fn get(&self) -> T {
        Default::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MyType {
    dummy: usize
}

impl MyTrait for MyType {
    fn get(&self) -> MyType { (*self).clone() }
}

fn test_eq<M>(m: M, n: M)
where M : MyTrait + Debug + PartialEq
{
    assert_eq!(m.get(), n);
}

pub fn main() {
    test_eq(0_usize, 0_usize);

    let value = MyType { dummy: 256 + 22 };
    test_eq(value, value);
}
