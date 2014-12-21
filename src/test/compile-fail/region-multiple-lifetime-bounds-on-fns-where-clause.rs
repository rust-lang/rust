// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn a<'a, 'b, 'c>(x: &mut &'a int, y: &mut &'b int, z: &mut &'c int) where 'b: 'a + 'c {
    // Note: this is legal because of the `'b:'a` declaration.
    *x = *y;
    *z = *y;
}

fn b<'a, 'b, 'c>(x: &mut &'a int, y: &mut &'b int, z: &mut &'c int) {
    // Illegal now because there is no `'b:'a` declaration.
    *x = *y; //~ ERROR cannot infer
    *z = *y; //~ ERROR cannot infer
}

fn c<'a,'b, 'c>(x: &mut &'a int, y: &mut &'b int, z: &mut &'c int) {
    // Here we try to call `foo` but do not know that `'a` and `'b` are
    // related as required.
    a(x, y, z); //~ ERROR cannot infer
}

fn d() {
    // 'a and 'b are early bound in the function `a` because they appear
    // inconstraints:
    let _: fn(&mut &int, &mut &int, &mut &int) = a; //~ ERROR mismatched types
}

fn e() {
    // 'a and 'b are late bound in the function `b` because there are
    // no constraints:
    let _: fn(&mut &int, &mut &int, &mut &int) = b;
}

fn main() { }
