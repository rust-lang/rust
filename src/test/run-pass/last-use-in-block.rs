// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #1818

fn lp<T>(s: ~str, f: &fn(~str) -> T) -> T {
    while false {
        let r = f(s);
        return (r);
    }
    fail2!();
}

fn apply<T>(s: ~str, f: &fn(~str) -> T) -> T {
    fn g<T>(s: ~str, f: &fn(~str) -> T) -> T {f(s)}
    g(s, |v| { let r = f(v); r })
}

pub fn main() {}
