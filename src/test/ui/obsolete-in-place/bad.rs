// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `<-` and `in` syntax gets a hard error.

// revisions: good bad
//[good] run-pass

#[cfg(bad)]
fn main() {
    let (x, y, foo, bar);
    x <- y; //[bad]~ ERROR emplacement syntax is obsolete
    in(foo) { bar }; //[bad]~ ERROR emplacement syntax is obsolete
}

#[cfg(good)]
fn main() {
}
