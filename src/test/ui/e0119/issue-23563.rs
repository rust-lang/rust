// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_23563_a.rs

// Ref: https://github.com/rust-lang/rust/issues/23563#issuecomment-260751672

extern crate issue_23563_a as a;

use a::LolFrom;
use a::LolInto;
use a::LolTo;

struct LocalType<T>(Option<T>);

impl<'a, T> LolFrom<&'a [T]> for LocalType<T> { //~ ERROR conflicting implementations of trait
    fn from(_: &'a [T]) -> LocalType<T> { LocalType(None) }
}

impl<T> LolInto<LocalType<T>> for LocalType<T> {
    fn convert_into(self) -> LocalType<T> {
        self
    }
}

impl LolTo<LocalType<u8>> for [u8] {
    fn convert_to(&self) -> LocalType<u8> {
        LocalType(None)
    }
}

fn main() {}
