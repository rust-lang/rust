#![crate_type = "rlib"]

#[link(name = "foo", kind = "static")]
extern "C" {
    fn foo();
}

pub fn doit() {
    unsafe {
        foo();
    }
}
