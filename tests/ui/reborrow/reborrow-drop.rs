//@ compile-flags: --crate-type=lib

#![feature(reborrow)]

use std::marker::Reborrow;

struct MyMut<'a>(&'a mut ());

impl Reborrow for MyMut<'_> {}
//~^ ERROR the trait `Reborrow` cannot be implemented for this type; the type has a destructor [E0184]

impl Drop for MyMut<'_> {
    fn drop(&mut self) {}
}
