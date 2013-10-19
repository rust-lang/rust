// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[linkage(external)]
#[inline(always)] // Force LLVM to consider the function for removal.
#[no_mangle]
fn foobar() {}

// The following tests call foobar directly via inline assembly, using bl on ARM and call on x86.
// This is done so that LLVM's decisions aren't affected by the reference to foobar, and without
// #[linkage(external)], LLVM fails with the error "undefined reference to `foobar'".

#[cfg(target_arch = "arm")]
pub fn main() {
    foobar();
    unsafe { asm!("bl foobar"); }
}

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "x86_64")]
pub fn main() {
    foobar();
    unsafe { asm!("call foobar"); }
}

#[cfg(not(target_arch = "arm"), not(target_arch = "x86"), not(target_arch = "x86_64"))]
pub fn main() {}
