// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test #5723

// Test that you cannot escape a borrowed pointer
// into a trait.

struct ctxt { v: uint }

trait get_ctxt {
    fn get_ctxt(&self) -> &'self ctxt;
}

struct has_ctxt<'self> { c: &'self ctxt }

impl<'self> get_ctxt for has_ctxt<'self> {
    fn get_ctxt(&self) -> &'self ctxt { self.c }
}

fn make_gc() -> @get_ctxt  {
    let ctxt = ctxt { v: 22u };
    let hc = has_ctxt { c: &ctxt };
    return @hc as @get_ctxt;
    //^~ ERROR source contains borrowed pointer
}

fn main() {
    make_gc().get_ctxt().v;
}
