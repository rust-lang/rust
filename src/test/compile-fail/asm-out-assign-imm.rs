// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast #[feature] doesn't work with check-fast
#[feature(asm)];

fn foo(x: int) { info2!("{}", x); }

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "x86_64")]
pub fn main() {
    let x: int;
    x = 1; //~ NOTE prior assignment occurs here
    foo(x);
    unsafe {
        asm!("mov $1, $0" : "=r"(x) : "r"(5u)); //~ ERROR re-assignment of immutable variable `x`
    }
    foo(x);
}

#[cfg(not(target_arch = "x86"), not(target_arch = "x86_64"))]
pub fn main() {}
