// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: mismatched types
extern mod std;
use std::map::HashMap;
use std::bitv;

type fn_info = {vars: HashMap<uint, var_info>};
type var_info = {a: uint, b: uint};

fn bitv_to_str(enclosing: fn_info, v: ~bitv::Bitv) -> str {
    let s = "";

    // error is that the value type in the hash map is var_info, not a box
    for enclosing.vars.each_value |val| {
        if v.get(val) { s += "foo"; }
    }
    return s;
}

fn main() { debug!("OK"); }
