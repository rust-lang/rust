// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-F private_no_mangle_fns -F no_mangle_const_items -F private_no_mangle_statics

#[no_mangle]
fn foo() { //~ ERROR function is marked #[no_mangle], but not exported
}

#[allow(dead_code)]
#[no_mangle]
const FOO: u64 = 1; //~ ERROR const items should never be #[no_mangle]

#[no_mangle]
pub const PUB_FOO: u64 = 1; //~ ERROR const items should never be #[no_mangle]

#[no_mangle]
pub fn bar()  {
}

#[no_mangle]
pub static BAR: u64 = 1;

#[allow(dead_code)]
#[no_mangle]
static PRIVATE_BAR: u64 = 1; //~ ERROR static is marked #[no_mangle], but not exported


fn main() {
    foo();
    bar();
}
