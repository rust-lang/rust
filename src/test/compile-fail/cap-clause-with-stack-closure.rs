// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(_f: fn()) {}
fn bar(_f: @int) {}

fn main() {
    let x = @3;
    foo(|| bar(x) );

    let x = @3;
    foo(|copy x| bar(x) ); //~ ERROR cannot capture values explicitly with a block closure

    let x = @3;
    foo(|move x| bar(x) ); //~ ERROR cannot capture values explicitly with a block closure
}

