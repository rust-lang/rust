// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



fn g<X>(x: X) -> X { return x; }

#[deriving(Clone)]
struct Pair<T> {
    a: T,
    b: T
}

fn f<T:Clone>(t: T) -> Pair<T> {
    let x: Pair<T> = Pair {a: t.clone(), b: t};
    return g::<Pair<T>>(x);
}

pub fn main() {
    let b = f::<int>(10);
    info!("{:?}" ,b.a);
    info!("{:?}", b.b);
    assert_eq!(b.a, 10);
    assert_eq!(b.b, 10);
}
