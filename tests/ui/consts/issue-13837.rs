//@ check-pass
#![allow(dead_code)]

struct TestStruct {
    x: *const [isize; 2]
}

unsafe impl Sync for TestStruct {}

static TEST_VALUE : TestStruct = TestStruct{x: 0x1234 as *const [isize; 2]};

fn main() {}
