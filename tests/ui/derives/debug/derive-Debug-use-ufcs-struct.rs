//@ run-pass
#![allow(warnings)]

#[derive(Debug)]
pub struct Bar { pub t: () }

impl<T> Access for T {}
pub trait Access {
    fn field(&self, _: impl Sized, _: impl Sized) {
        panic!("got into Access::field");
    }

    fn finish(&self) -> Result<(), std::fmt::Error> {
        panic!("got into Access::finish");
    }

    fn debug_struct(&self, _: impl Sized, _: impl Sized) {
        panic!("got into Access::debug_struct");
    }
}

impl<T> MutAccess for T {}
pub trait MutAccess {
    fn field(&mut self, _: impl Sized, _: impl Sized) {
        panic!("got into MutAccess::field");
    }

    fn finish(&mut self) -> Result<(), std::fmt::Error> {
        panic!("got into MutAccess::finish");
    }

    fn debug_struct(&mut self, _: impl Sized, _: impl Sized) {
        panic!("got into MutAccess::debug_struct");
    }
}

fn main() {
    let bar = Bar { t: () };
    assert_eq!("Bar { t: () }", format!("{:?}", bar));
}
