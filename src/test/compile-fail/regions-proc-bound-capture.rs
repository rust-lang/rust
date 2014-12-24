// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn borrowed_proc<'a>(x: &'a int) -> Box<FnMut()->(int) + 'a> {
    // This is legal, because the region bound on `proc`
    // states that it captures `x`.
    box move|| { *x }
}

fn static_proc(x: &int) -> Box<FnMut()->(int) + 'static> {
    // This is illegal, because the region bound on `proc` is 'static.
    box move|| { *x } //~ ERROR cannot infer
}

fn main() { }
