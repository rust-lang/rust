// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that shapes respect linearize_ty_params().

enum option<T> {
    none,
    some(T),
}

struct Smallintmap<T> {v: ~[option<T>]}

struct V<T> { v: ~[option<T>] }

fn mk<T:'static>() -> @mut Smallintmap<T> {
    let v: ~[option<T>] = ~[];
    return @mut Smallintmap {v: v};
}

fn f<T,U:'static>() {
    let sim = mk::<U>();
    error2!("{:?}", sim);
}

pub fn main() {
    f::<int,int>();
}
