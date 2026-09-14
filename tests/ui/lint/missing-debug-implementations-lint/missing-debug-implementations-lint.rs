//! Test the `missing_debug_implementations` lint that warns about public types without Debug.
//!
//! See https://github.com/rust-lang/rust/issues/20855

//@ compile-flags: --crate-type lib
#![deny(missing_debug_implementations)]
#![allow(unused)]

use std::fmt;

pub enum A {} //~ ERROR type does not implement `Debug`

#[derive(Debug)]
pub enum B {}

pub enum C {}
impl fmt::Debug for C {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

pub struct Foo; //~ ERROR type does not implement `Debug`

#[derive(Debug)]
pub struct Bar;

pub struct Baz;
impl fmt::Debug for Baz {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

// Private types should not trigger the lint
struct PrivateStruct;
enum PrivateEnum {}

#[derive(Debug)]
pub struct GenericType<T>(T);
