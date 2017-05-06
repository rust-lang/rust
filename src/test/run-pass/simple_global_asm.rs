// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(global_asm)]
#![feature(naked_functions)]

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
global_asm!(r#"
    .global foo
    .global _foo
foo:
_foo:
    ret
"#);

extern {
    fn foo();
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn main() { unsafe { foo(); } }

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
fn main() {}
