#![allow(unused_parens)]
// aux-build:cond_plugin.rs

#![feature(proc_macro_hygiene)]

extern crate cond_plugin;

use cond_plugin::cond;

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
