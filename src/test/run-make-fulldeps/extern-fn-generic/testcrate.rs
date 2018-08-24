#![crate_type = "lib"]

#[repr(C)]
pub struct TestStruct<T> {
    pub x: u8,
    pub y: T
}

pub extern "C" fn foo<T>(ts: TestStruct<T>) -> T { ts.y }

#[link(name = "test", kind = "static")]
extern {
    pub fn call(c: extern "C" fn(TestStruct<i32>) -> i32) -> i32;
}
