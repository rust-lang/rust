// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum foo = {mut bar: baz};

enum baz = @{mut baz: int};

trait frob {
    fn frob();
}

impl foo: frob {
    fn frob() {
        really_impure(self.bar);
    }
}

// Override default mode so that we are passing by value
fn really_impure(++bar: baz) {
    bar.baz = 3;
}

fn main() {}
