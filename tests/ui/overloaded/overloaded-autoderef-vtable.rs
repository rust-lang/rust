// run-pass
#![allow(dead_code)]

use std::ops::Deref;

struct DerefWithHelper<H, T> {
    helper: H,
    value: T
}

trait Helper<T> {
    fn helper_borrow(&self) -> &T;
}

impl<T> Helper<T> for Option<T> {
    fn helper_borrow(&self) -> &T {
        self.as_ref().unwrap()
    }
}

impl<T, H: Helper<T>> Deref for DerefWithHelper<H, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.helper.helper_borrow()
    }
}

struct Foo {x: isize}

impl Foo {
    fn foo(&self) -> isize {self.x}
}

pub fn main() {
    let x: DerefWithHelper<Option<Foo>, Foo> =
        DerefWithHelper { helper: Some(Foo {x: 5}), value: Foo { x: 2 } };
    assert_eq!(x.foo(), 5);
}
