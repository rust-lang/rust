// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that the spans of import errors are correct.

use abc::one_el;
//~^ ERROR 13:5: 13:16
use abc::{a, bbb, cccccc};
//~^ ERROR 15:11: 15:12
//~^^ ERROR 15:14: 15:17
//~^^^ ERROR 15:19: 15:25
use a_very_long_name::{el, el2};
//~^ ERROR 19:24: 19:26
//~^^ ERROR 19:28: 19:31

fn main() {}
