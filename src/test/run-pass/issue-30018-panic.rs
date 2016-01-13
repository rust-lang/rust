// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for Issue #30018. This is very similar to the
// original reported test, except that the panic is wrapped in a
// spawned thread to isolate the expected error result from the
// SIGTRAP injected by the drop-flag consistency checking.

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn foo() -> Foo {
    panic!();
}

fn main() {
    use std::thread;
    let handle = thread::spawn(|| {
        let _ = &[foo()];
    });
    let _ = handle.join();
}
