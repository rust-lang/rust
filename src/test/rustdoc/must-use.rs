// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has must_use/struct.Struct.html //pre '#[must_use]'
#[must_use]
pub struct Struct {
    field: i32,
}

// @has must_use/enum.Enum.html //pre '#[must_use = "message"]'
#[must_use = "message"]
pub enum Enum {
    Variant(i32),
}
