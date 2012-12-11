// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait get_ctxt {
    fn get_ctxt() -> &self/uint;
}

fn make_gc1(gc: get_ctxt/&a) -> get_ctxt/&b  {
    return gc; //~ ERROR mismatched types: expected `@get_ctxt/&b` but found `@get_ctxt/&a`
}

fn make_gc2(gc: get_ctxt/&a) -> get_ctxt/&b  {
    return gc as get_ctxt; //~ ERROR cannot infer an appropriate lifetime
}

fn main() {
}
