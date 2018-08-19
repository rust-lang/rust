// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_raw_ptr_to_usize_cast, const_compare_raw_pointers, const_raw_ptr_deref)]

fn main() {}

// unconst and bad, will thus error in miri
const X: bool = &1 as *const i32 == &2 as *const i32; //~ ERROR cannot be used
// unconst and fine
const X2: bool = 42 as *const i32 == 43 as *const i32;
// unconst and fine
const Y: usize = 42usize as *const i32 as usize + 1;
// unconst and bad, will thus error in miri
const Y2: usize = &1 as *const i32 as usize + 1; //~ ERROR cannot be used
// unconst and fine
const Z: i32 = unsafe { *(&1 as *const i32) };
// unconst and bad, will thus error in miri
const Z2: i32 = unsafe { *(42 as *const i32) }; //~ ERROR cannot be used
const Z3: i32 = unsafe { *(44 as *const i32) }; //~ ERROR cannot be used
