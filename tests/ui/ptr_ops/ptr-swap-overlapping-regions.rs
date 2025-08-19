//! Check that `std::ptr::swap` behaves correctly when the source and destination
//! pointers refer to the same memory location, avoiding issues like overlapping `memcpy`.
//!
//! Regression test: <https://github.com/rust-lang/rust/issues/5041>

//@ run-pass

#![allow(dead_code)]

use std::ptr;

pub fn main() {
    let mut test = TestDescAndFn {
        desc: TestDesc { name: TestName::DynTestName("test".to_string()), should_fail: false },
        testfn: TestFn::DynTestFn(22),
    };
    do_swap(&mut test);
}

fn do_swap(test: &mut TestDescAndFn) {
    unsafe {
        ptr::swap(test, test);
    }
}

pub enum TestName {
    DynTestName(String),
}

pub enum TestFn {
    DynTestFn(isize),
    DynBenchFn(isize),
}

pub struct TestDesc {
    name: TestName,
    should_fail: bool,
}

pub struct TestDescAndFn {
    desc: TestDesc,
    testfn: TestFn,
}
