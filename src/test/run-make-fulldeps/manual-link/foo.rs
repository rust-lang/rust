#![crate_type = "rlib"]

extern {
    fn bar();
}

pub fn foo() {
    unsafe { bar(); }
}
