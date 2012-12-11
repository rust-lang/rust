// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type foo = { mut z : fn@() };

fn nop() { }
fn nop_foo(_y: @int, _x : @foo) { }

fn o() -> @int { @10 }

fn main() {
    let w = @{ mut z: {||nop()} };
    let x = {||nop_foo(o(), w)};
    w.z = x;
}