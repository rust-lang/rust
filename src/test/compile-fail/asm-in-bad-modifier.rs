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
    let y: int;
    unsafe {
        asm!("mov $1, $0" : "=r"(x) : "=r"(5u)); //~ ERROR input operand constraint contains '='
        asm!("mov $1, $0" : "=r"(y) : "+r"(5u)); //~ ERROR input operand constraint contains '+'
    }
    foo(x);
    foo(y);
}

#[cfg(not(target_arch = "x86"), not(target_arch = "x86_64"))]
pub fn main() {}
