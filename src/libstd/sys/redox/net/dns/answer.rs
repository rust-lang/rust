// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use string::String;
use vec::Vec;

#[derive(Clone, Debug)]
pub struct DnsAnswer {
    pub name: String,
    pub a_type: u16,
    pub a_class: u16,
    pub ttl_a: u16,
    pub ttl_b: u16,
    pub data: Vec<u8>
}
