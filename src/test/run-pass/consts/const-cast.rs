// run-pass
#![allow(non_upper_case_globals)]

struct TestStruct {
    x: *const u8,
}

unsafe impl Sync for TestStruct {}

extern fn foo() {}
const x: extern "C" fn() = foo;
static y: TestStruct = TestStruct { x: x as *const u8 };

pub fn main() {
    assert_eq!(x as *const u8, y.x);
}
