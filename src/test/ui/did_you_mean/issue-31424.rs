// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// forbid-output: &mut mut self

struct Struct;

impl Struct {
    fn foo(&mut self) {
        (&mut self).bar();
    }

    // In this case we could keep the suggestion, but to distinguish the
    // two cases is pretty hard. It's an obscure case anyway.
    fn bar(self: &mut Self) {
        (&mut self).bar();
    }
}

fn main () {}
