// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(existential_type)]

fn main() {
}

existential type PartiallyDefined<T>: 'static; //~ `T` is unused

fn partially_defined<T: std::fmt::Debug>(_: T) -> PartiallyDefined<T> {
    4u32
}

existential type PartiallyDefined2<T>: 'static; //~ `T` is unused

fn partially_defined2<T: std::fmt::Debug>(_: T) -> PartiallyDefined2<T> {
    4u32
}

fn partially_defined22<T>(_: T) -> PartiallyDefined2<T> {
    4u32
}
