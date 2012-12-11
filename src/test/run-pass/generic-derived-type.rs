// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



fn g<X: Copy>(x: X) -> X { return x; }

fn f<T: Copy>(t: T) -> {a: T, b: T} {
    type pair = {a: T, b: T};

    let x: pair = {a: t, b: t};
    return g::<pair>(x);
}

fn main() {
    let b = f::<int>(10);
    log(debug, b.a);
    log(debug, b.b);
    assert (b.a == 10);
    assert (b.b == 10);
}
