// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(shadowed_primitive_type_names)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

type bool = (); //~ ERROR shadows a built-in primitive type

struct char; //~ ERROR shadows a built-in primitive type

enum int { //~ ERROR shadows a built-in primitive type
    Unused
}

trait u32 {} //~ ERROR shadows a built-in primitive type

fn main() { }
