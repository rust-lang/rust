// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A zero-dependency test that covers some basic traits, default
// methods, etc.  When mucking about with basic type system stuff I
// often encounter problems in the iterator trait, so it's useful to
// have hanging around. -nmatsakis

// error-pattern: requires `start` lang_item

#![no_std]
#![feature(lang_items)]

#[lang = "sized"]
pub trait Sized for Sized? {
    // Empty.
}

pub mod std {
    pub mod clone {
        pub trait Clone {
            fn clone(&self) -> Self;
        }
    }
}

pub struct ContravariantLifetime<'a>;

impl <'a> ::std::clone::Clone for ContravariantLifetime<'a> {
    #[inline]
    fn clone(&self) -> ContravariantLifetime<'a> {
        match *self { ContravariantLifetime => ContravariantLifetime, }
    }
}

fn main() { }
