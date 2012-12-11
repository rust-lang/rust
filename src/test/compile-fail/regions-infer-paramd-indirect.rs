// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we correctly infer that b and c must be region
// parameterized because they reference a which requires a region.

type a = &int;
type b = @a;
type c = {f: @b};

trait set_f {
    fn set_f_ok(b: @b/&self);
    fn set_f_bad(b: @b);
}

impl c: set_f {
    fn set_f_ok(b: @b/&self) {
        self.f = b;
    }

    fn set_f_bad(b: @b) {
        self.f = b; //~ ERROR mismatched types: expected `@@&self/int` but found `@@&int`
    }
}

fn main() {}
