// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate alloc; //~ ERROR: use of unstable library feature
extern crate arena; //~ ERROR: use of unstable library feature
extern crate collections; //~ ERROR: use of unstable library feature
extern crate core; //~ ERROR: use of unstable library feature
extern crate flate; //~ ERROR: use of unstable library feature
extern crate fmt_macros; //~ ERROR: use of unstable library feature
extern crate getopts; //~ ERROR: use of unstable library feature
extern crate graphviz; //~ ERROR: use of unstable library feature
extern crate libc; //~ ERROR: use of unstable library feature
extern crate log; //~ ERROR: use of unstable library feature
extern crate rand; //~ ERROR: use of unstable library feature
extern crate rbml; //~ ERROR: use of unstable library feature
extern crate rustc; //~ ERROR: use of unstable library feature
extern crate rustc_driver; //~ ERROR: use of unstable library feature
extern crate rustc_serialize; //~ ERROR: use of unstable library feature
extern crate rustdoc; //~ ERROR: use of unstable library feature
extern crate syntax; //~ ERROR: use of unstable library feature
extern crate term; //~ ERROR: use of unstable library feature
extern crate test; //~ ERROR: use of unstable library feature

mod foo {
    extern crate std;
}

fn main() {}
