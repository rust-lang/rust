//! Regression test for <https://github.com/rust-lang/rust/issues/148861>

//@ check-pass

#![deny(dead_code)]

use std::any::Any;

pub trait Foo: Any {}

impl Foo for Box<dyn Foo> {}

impl dyn Foo {
    #[expect(dead_code)]
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        let this = self as &dyn Any;
        if let Some(boxed) = this.downcast_ref::<Box<dyn Foo>>() {
            boxed.downcast_ref::<T>()
        } else {
            this.downcast_ref::<T>()
        }
    }
}

fn main() {}
