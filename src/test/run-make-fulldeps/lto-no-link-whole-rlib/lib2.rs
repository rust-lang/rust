#![crate_type = "rlib"]

extern crate lib1;

#[link(name = "bar", kind = "static")]
extern {
    fn foo() -> i32;
}

pub fn foo2() -> i32 {
    unsafe { foo() }
}
