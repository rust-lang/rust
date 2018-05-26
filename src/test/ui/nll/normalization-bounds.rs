// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that lifetime bounds get checked the right way around with NLL enabled.

//run-pass

#![feature(nll)]
trait Visitor<'d> {
    type Value;
}

impl<'a, 'd: 'a> Visitor<'d> for &'a () {
    type Value = ();
}

fn visit_seq<'d: 'a, 'a>() -> <&'a () as Visitor<'d>>::Value {}

fn main() {}
