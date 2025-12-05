#![crate_type = "rlib"]

extern "C" {
    fn bar();
}

pub fn foo() {
    unsafe {
        bar();
    }
}
