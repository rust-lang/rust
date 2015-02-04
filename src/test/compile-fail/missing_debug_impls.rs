// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --crate-type lib
#![deny(missing_debug_implementations)]
#![allow(unused)]

use std::fmt;

pub enum A {} //~ ERROR type does not implement `fmt::Debug`

#[derive(Debug)]
pub enum B {}

pub enum C {}

impl fmt::Debug for C {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

pub struct Foo; //~ ERROR type does not implement `fmt::Debug`

#[derive(Debug)]
pub struct Bar;

pub struct Baz;

impl fmt::Debug for Baz {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

struct PrivateStruct;

enum PrivateEnum {}

#[derive(Debug)]
struct GenericType<T>(T);
