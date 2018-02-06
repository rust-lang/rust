// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn cplusplus_mode_exceptionally_unsafe(x: &mut Option<&'static mut isize>) {
    let mut z = (0, 0);
    *x = Some(&mut z.1);
    //[ast]~^ ERROR `z.1` does not live long enough [E0597]
    //[mir]~^^ ERROR `z.1` does not live long enough [E0597]
    panic!("catch me for a dangling pointer!")
}

fn main() {
    cplusplus_mode_exceptionally_unsafe(&mut None);
}
