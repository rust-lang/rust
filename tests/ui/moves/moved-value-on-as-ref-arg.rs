//@ run-rustfix
#![allow(unused_mut)]
use std::borrow::{Borrow, BorrowMut};
use std::convert::{AsMut, AsRef};
struct Bar;

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

fn foo<T: AsRef<Bar>>(_: T) {}
fn qux<T: AsMut<Bar>>(_: T) {}
fn bat<T: Borrow<Bar>>(_: T) {}
fn baz<T: BorrowMut<Bar>>(_: T) {}

pub fn main() {
    let bar = Bar;
    foo(bar);
    let _baa = bar; //~ ERROR use of moved value
    let mut bar = Bar;
    qux(bar);
    let _baa = bar; //~ ERROR use of moved value
    let bar = Bar;
    bat(bar);
    let _baa = bar; //~ ERROR use of moved value
    let mut bar = Bar;
    baz(bar);
    let _baa = bar; //~ ERROR use of moved value
}
