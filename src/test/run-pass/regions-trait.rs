// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Ctxt { v: uint }

trait get_ctxt<'self> {
    fn get_ctxt(&self) -> &'self Ctxt;
}

struct HasCtxt<'self> { c: &'self Ctxt }

impl<'self> get_ctxt<'self> for HasCtxt<'self> {
    fn get_ctxt(&self) -> &'self Ctxt {
        self.c
    }
}

fn get_v(gc: @get_ctxt) -> uint {
    gc.get_ctxt().v
}

pub fn main() {
    let ctxt = Ctxt { v: 22 };
    let hc = HasCtxt { c: &ctxt };

    assert_eq!(get_v(@hc as @get_ctxt), 22);
}
