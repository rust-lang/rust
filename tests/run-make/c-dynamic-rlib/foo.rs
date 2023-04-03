#![crate_type = "rlib"]

#[link(name = "cfoo")]
extern "C" {
    fn foo();
}

pub fn rsfoo() {
    unsafe { foo() }
}
