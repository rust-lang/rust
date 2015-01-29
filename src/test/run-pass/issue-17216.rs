// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

struct Leak<'a> {
    dropped: &'a mut bool
}

#[unsafe_destructor]
impl<'a> Drop for Leak<'a> {
    fn drop(&mut self) {
        *self.dropped = true;
    }
}

fn main() {
    let mut dropped = false;
    {
        let leak = Leak { dropped: &mut dropped };
        // FIXME(#21721) "hack" used to be () but that can cause
        // certain LLVM versions to abort during optimizations.
        for (_, leaked) in Some(("hack", leak)).into_iter() {}
    }

    assert!(dropped);
}
