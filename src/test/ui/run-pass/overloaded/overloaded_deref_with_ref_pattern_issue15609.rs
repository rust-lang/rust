// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we choose Deref or DerefMut appropriately based on mutability of ref bindings (#15609).

fn main() {
    use std::cell::RefCell;

    struct S {
        node: E,
    }

    enum E {
        Foo(u32),
        Bar,
    }

    // Check match
    let x = RefCell::new(S { node: E::Foo(0) });

    let mut b = x.borrow_mut();
    match b.node {
        E::Foo(ref mut n) => *n += 1,
        _ => (),
    }

    // Check let
    let x = RefCell::new(0);
    let mut y = x.borrow_mut();
    let ref mut z = *y;

    fn foo(a: &mut RefCell<Option<String>>) {
        if let Some(ref mut s) = *a.borrow_mut() {
            s.push('a')
        }
    }
}
