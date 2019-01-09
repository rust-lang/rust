#![allow(dead_code)]
// Issue #5041 - avoid overlapping memcpy when src and dest of a swap are the same

// pretty-expanded FIXME #23616

use std::ptr;

pub fn main() {
    let mut test = TestDescAndFn {
        desc: TestDesc {
            name: TestName::DynTestName("test".to_string()),
            should_fail: false
        },
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
    DynTestName(String)
}

pub enum TestFn {
    DynTestFn(isize),
    DynBenchFn(isize),
}

pub struct TestDesc {
    name: TestName,
    should_fail: bool
}

pub struct TestDescAndFn {
    desc: TestDesc,
    testfn: TestFn,
}
