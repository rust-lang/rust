#![crate_type = "rlib"]

#[link(name = "foo", kind = "static")]
extern {
    fn foo() -> i32;
}

pub fn foo1() -> i32 {
    unsafe { foo() }
}
