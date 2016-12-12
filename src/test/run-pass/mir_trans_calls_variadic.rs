// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_interesting_average(_: i64, ...) -> f64;
}

fn test<T, U>(a: i64, b: i64, c: i64, d: i64, e: i64, f: T, g: U) -> i64 {
    unsafe {
        rust_interesting_average(6, a, a as f64,
                                    b, b as f64,
                                    c, c as f64,
                                    d, d as f64,
                                    e, e as f64,
                                    f, g) as i64
    }
}

fn main(){
    assert_eq!(test(10, 20, 30, 40, 50, 60_i64, 60.0_f64), 70);
}
