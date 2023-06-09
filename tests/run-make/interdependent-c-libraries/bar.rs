#![crate_type = "rlib"]

extern crate foo;

#[link(name = "bar", kind = "static")]
extern "C" {
    fn bar();
}

pub fn doit() {
    unsafe {
        bar();
    }
}
