// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test #5723

// Test that you cannot escape a reference
// into a trait.

struct ctxt { v: uint }

trait get_ctxt {
    fn get_ctxt(&self) -> &'a ctxt;
}

struct has_ctxt<'a> { c: &'a ctxt }

impl<'a> get_ctxt for has_ctxt<'a> {
    fn get_ctxt(&self) -> &'a ctxt { self.c }
}

fn make_gc() -> @get_ctxt  {
    let ctxt = ctxt { v: 22u };
    let hc = has_ctxt { c: &ctxt };
    return @hc as @get_ctxt;
    //~^ ERROR source contains reference
}

fn main() {
    make_gc().get_ctxt().v;
}
