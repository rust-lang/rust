// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

enum t { make_t(@int), clam, }

fn foo(s: @int) {
    println!("{:?}", ::std::managed::refcount(s));
    let count = ::std::managed::refcount(s);
    let x: t = make_t(s); // ref up
    assert_eq!(::std::managed::refcount(s), count + 1u);
    println!("{:?}", ::std::managed::refcount(s));

    match x {
      make_t(y) => {
        println!("{:?}", y); // ref up then down

      }
      _ => { println!("?"); fail!(); }
    }
    println!("{:?}", ::std::managed::refcount(s));
    assert_eq!(::std::managed::refcount(s), count + 1u);
    let _ = ::std::managed::refcount(s); // don't get bitten by last-use.
}

pub fn main() {
    let s: @int = @0; // ref up

    let count = ::std::managed::refcount(s);

    foo(s); // ref up then down

    println!("{}", ::std::managed::refcount(s));
    let count2 = ::std::managed::refcount(s);
    assert_eq!(count, count2);
}
