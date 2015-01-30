// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-F private_no_mangle_fns

// FIXME(#19495) no_mangle'ing main ICE's.
#[no_mangle]
fn foo() { //~ ERROR function foo is marked #[no_mangle], but not exported
}

#[no_mangle]
pub fn bar()  {
}

fn main() {
    foo();
    bar();
}
