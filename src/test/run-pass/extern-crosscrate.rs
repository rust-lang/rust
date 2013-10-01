// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
//aux-build:extern-crosscrate-source.rs

extern mod externcallback(vers = "0.1");

#[fixed_stack_segment] #[inline(never)]
fn fact(n: uint) -> uint {
    unsafe {
        info2!("n = {}", n);
        externcallback::rustrt::rust_dbg_call(externcallback::cb, n)
    }
}

pub fn main() {
    let result = fact(10u);
    info2!("result = {}", result);
    assert_eq!(result, 3628800u);
}
