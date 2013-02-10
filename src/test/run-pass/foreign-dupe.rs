// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast - Somehow causes check-fast to livelock?? Probably because we're
// calling pin_task and that's having wierd side-effects.

#[abi = "cdecl"]
#[link_name = "rustllvm"]
extern mod rustllvm1 {
    pub unsafe fn LLVMGetLastError() -> *libc::c_char;
}

#[abi = "cdecl"]
#[link_name = "rustllvm"]
extern mod rustllvm2 {
    pub unsafe fn LLVMGetLastError() -> *libc::c_char;
}

pub fn main() {
    unsafe {
        rustllvm1::LLVMGetLastError();
        rustllvm2::LLVMGetLastError();
    }
}
