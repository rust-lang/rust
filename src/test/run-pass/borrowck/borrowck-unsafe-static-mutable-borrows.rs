// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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

// Test file taken from issue 45129 (https://github.com/rust-lang/rust/issues/45129)

struct Foo { x: [usize; 2] }

static mut SFOO: Foo = Foo { x: [23, 32] };

impl Foo {
    fn x(&mut self) -> &mut usize { &mut self.x[0] }
}

fn main() {
    unsafe {
        let sfoo: *mut Foo = &mut SFOO;
        let x = (*sfoo).x();
        (*sfoo).x[1] += 1;
        *x += 1;
    }
}
