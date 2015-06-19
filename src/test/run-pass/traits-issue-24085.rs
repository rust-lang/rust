// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that trait selection for Copy does not introduce
// bad region constraints.  Issue #24085.

#[derive(Clone, Copy)]
enum Path<'a:'b, 'b> {
    Data(&'a i32),
    Link(&'a i32, &'b Path<'a, 'b>)
}

fn foo<'a,'b,F>(_p: Path<'a, 'b>, _f: F)
                where F: for<'c> FnMut(&Path<'a, 'c>) {
}

fn main() {
    let y = 0;
    let p = Path::Data(&y);

    foo(p, |x| {*x;});
}
