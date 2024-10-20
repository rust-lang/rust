//! auxiliary definitons for suggest-borrow-for-generic-arg.rs, to ensure the suggestion works on
//! functions defined in other crates.

use std::borrow::{Borrow, BorrowMut};
use std::convert::{AsMut, AsRef};
pub struct Bar;

impl AsRef<Bar> for Bar {
    fn as_ref(&self) -> &Bar {
        self
    }
}

impl AsMut<Bar> for Bar {
    fn as_mut(&mut self) -> &mut Bar {
        self
    }
}

pub fn foo<T: AsRef<Bar>>(_: T) {}
pub fn qux<T: AsMut<Bar>>(_: T) {}
pub fn bat<T: Borrow<T>>(_: T) {}
pub fn baz<T: BorrowMut<T>>(_: T) {}
