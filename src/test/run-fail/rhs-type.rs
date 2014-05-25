// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that trans treats the rhs of pth's decl
// as a _|_-typed thing, not a str-typed thing
// error-pattern:bye

#![allow(unreachable_code)]
#![allow(unused_variable)]

struct T { t: String }

fn main() {
    let pth = fail!("bye");
    let _rs: T = T {t: pth};
}
