// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// In theory, it doesn't matter what order destructors are run in for rust
// because we have explicit ownership of values meaning that there's no need to
// run one before another. With unsafe code, however, there may be a safe
// interface which relies on fields having their destructors run in a particular
// order. At the time of this writing, std::rt::sched::Scheduler is an example
// of a structure which contains unsafe handles to FFI-like types, and the
// destruction order of the fields matters in the sense that some handles need
// to get destroyed before others.
//
// In C++, destruction order happens bottom-to-top in order of field
// declarations, but we currently run them top-to-bottom. I don't think the
// order really matters that much as long as we define what it is.

struct A;
struct B;
struct C {
    a: A,
    b: B,
}

static mut hit: bool = false;

impl Drop for A {
    fn drop(&mut self) {
        unsafe {
            assert!(!hit);
            hit = true;
        }
    }
}

impl Drop for B {
    fn drop(&mut self) {
        unsafe {
            assert!(hit);
        }
    }
}

pub fn main() {
    let _c = C { a: A, b: B };
}
