// run-pass
#![allow(non_camel_case_types)]

// Test that we pick which version of `foo` to run based on the
// type that is (ultimately) inferred for `x`.


trait foo {
    fn foo(&self) -> i32;
}

impl foo for Vec<u32> {
    fn foo(&self) -> i32 {1}
}

impl foo for Vec<i32> {
    fn foo(&self) -> i32 {2}
}

fn call_foo_uint() -> i32 {
    let mut x = Vec::new();
    let y = x.foo();
    x.push(0u32);
    y
}

fn call_foo_int() -> i32 {
    let mut x = Vec::new();
    let y = x.foo();
    x.push(0i32);
    y
}

fn main() {
    assert_eq!(call_foo_uint(), 1);
    assert_eq!(call_foo_int(), 2);
}
