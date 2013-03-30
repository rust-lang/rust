// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

struct Arg<T> {val: T, fin: extern fn(T)}

struct finish<T> {
  arg: Arg<T>
}

#[unsafe_destructor]
impl<T:Copy> Drop for finish<T> {
    fn finalize(&self) {
        unsafe {
            (self.arg.fin)(self.arg.val);
        }
    }
}

fn finish<T:Copy>(arg: Arg<T>) -> finish<T> {
    finish {
        arg: arg
    }
}

pub fn main() {
    let box = @mut 10;
    fn dec_box(i: @mut int) { *i -= 1; }

    { let _i = finish(Arg{val: box, fin: dec_box}); }
    assert!((*box == 9));
}
