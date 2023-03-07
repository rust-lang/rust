// run-pass
// Test case where an associated type is referenced from within the
// supertrait definition. Issue #20220.


use std::vec::IntoIter;

pub trait Foo: Iterator<Item=<Self as Foo>::Key> {
    type Key;
}

impl Foo for IntoIter<i32> {
    type Key = i32;
}

fn sum_foo<F:Foo<Key=i32>>(f: F) -> i32 {
    f.fold(0, |a,b| a + b)
}

fn main() {
    let x = sum_foo(vec![11, 10, 1].into_iter());
    assert_eq!(x, 22);
}
