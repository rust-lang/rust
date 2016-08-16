// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cond_plugin.rs
// ignore-stage1

#![feature(plugin)]
#![feature(rustc_private)]
#![plugin(cond_plugin)]

fn fact(n : i64) -> i64 {
    if n == 0 {
        1
    } else {
        n * fact(n - 1)
    }
}

fn fact_cond(n : i64) -> i64 {
  cond!(
    ((n == 0) 1)
    (else (n * fact_cond(n-1)))
  )
}

fn fib(n : i64) -> i64 {
  if n == 0 || n == 1 {
      1
  } else {
      fib(n-1) + fib(n-2)
  }
}

fn fib_cond(n : i64) -> i64 {
  cond!(
    ((n == 0) 1)
    ((n == 1) 1)
    (else (fib_cond(n-1) + fib_cond(n-2)))
  )
}

fn main() {
    assert_eq!(fact(3), fact_cond(3));
    assert_eq!(fact(5), fact_cond(5));
    assert_eq!(fib(5), fib_cond(5));
    assert_eq!(fib(8), fib_cond(8));
}
