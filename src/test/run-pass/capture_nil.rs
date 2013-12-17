// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z no-opt

// This test has to be setup just so to trigger
// the condition which was causing us a crash.
// The situation is that we are capturing a
// () value by ref.  We generally feel free,
// however, to substitute NULL pointers and
// undefined values for values of () type, and
// so this caused a segfault when we copied into
// the closure.
//
// The fix is just to not emit any actual loads
// or stores for copies of () type (which is of
// course preferable, as the value itself is
// irrelevant).

use std::task;

fn foo(x: ()) -> Port<()> {
    let (p, c) = Chan::<()>::new();
    do task::spawn() {
        c.send(x);
    }
    p
}

pub fn main() {
    foo(()).recv()
}
