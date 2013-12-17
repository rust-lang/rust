// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];
#[crate_id="req"];
// NOTE: remove after the next snapshot
#[link(name = "req")];
#[crate_type = "lib"];

extern mod extra;

use std::hashmap::HashMap;

pub type header_map = HashMap<~str, @mut ~[@~str]>;

// the unused ty param is necessary so this gets monomorphized
pub fn request<T>(req: &header_map) {
  let _x = (*((**req.get(&~"METHOD")).clone())[0u]).clone();
}
