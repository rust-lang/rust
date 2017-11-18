// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:private-inferred-type.rs

// error-pattern:type `fn() {ext::priv_fn}` is private
// error-pattern:static `PRIV_STATIC` is private
// error-pattern:type `ext::PrivEnum` is private
// error-pattern:type `fn() {<u8 as ext::PrivTrait>::method}` is private
// error-pattern:type `fn(u8) -> ext::PrivTupleStruct {ext::PrivTupleStruct::{{constructor}}}` is pr
// error-pattern:type `fn(u8) -> ext::PubTupleStruct {ext::PubTupleStruct::{{constructor}}}` is priv
// error-pattern:type `for<'r> fn(&'r ext::Pub<u8>) {<ext::Pub<u8>>::priv_method}` is private

#![feature(decl_macro)]

extern crate private_inferred_type as ext;

fn main() {
    ext::m!();
}
