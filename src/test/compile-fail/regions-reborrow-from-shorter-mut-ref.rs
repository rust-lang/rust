// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #8624. Tests that reborrowing the contents of an `&'b mut`
// pointer which is backed by another `&'a mut` can only be done
// for `'a` (which must be a sublifetime of `'b`).

fn copy_borrowed_ptr<'a, 'b>(p: &'a mut &'b mut int) -> &'b mut int {
    &mut **p //~ ERROR cannot infer
}

fn main() {
    let mut x = 1;
    let mut y = &mut x;
    let z = copy_borrowed_ptr(&mut y);
    *y += 1;
    *z += 1;
}
