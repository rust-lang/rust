// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(universal_impl_trait)]
#![feature(conservative_impl_trait)]
#![deny(non_camel_case_types)]

#[allow(dead_code)]
fn qqq(lol: impl Iterator<Item=u32>) -> impl Iterator<Item=u64> {
        lol.map(|x|x as u64)
}

fn main() {}
