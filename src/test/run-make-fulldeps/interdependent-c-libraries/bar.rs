#![crate_type = "rlib"]

extern crate foo;

#[link(name = "bar", kind = "static")]
extern {
    fn bar();
}

pub fn doit() {
    unsafe { bar(); }
}
