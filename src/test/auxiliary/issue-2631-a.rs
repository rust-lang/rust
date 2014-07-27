// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id="req"]
#![crate_type = "lib"]

use std::cell::RefCell;
use std::collections::HashMap;
use std::gc::Gc;

pub type header_map = HashMap<String, Gc<RefCell<Vec<Gc<String>>>>>;

// the unused ty param is necessary so this gets monomorphized
pub fn request<T>(req: &header_map) {
  let _x = (**((**req.get(&"METHOD".to_string())).clone()).borrow()
                                                          .clone()
                                                          .get(0)).clone();
}
