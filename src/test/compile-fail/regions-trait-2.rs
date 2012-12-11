// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type ctxt = { v: uint };

trait get_ctxt {
    fn get_ctxt() -> &self/ctxt;
}

type has_ctxt = { c: &ctxt };

impl has_ctxt: get_ctxt {
    fn get_ctxt() -> &self/ctxt { self.c }
}

fn make_gc() -> get_ctxt  {
    let ctxt = { v: 22u };
    let hc = { c: &ctxt }; //~ ERROR illegal borrow
    return hc as get_ctxt;
}

fn main() {
    make_gc().get_ctxt().v;
}
