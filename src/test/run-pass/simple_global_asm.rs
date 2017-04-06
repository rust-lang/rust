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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
global_asm!(r#"
    .global foo
foo:
    jmp baz
"#);

extern {
    fn foo();
}

#[no_mangle]
pub extern fn baz() {}

fn main() {}
