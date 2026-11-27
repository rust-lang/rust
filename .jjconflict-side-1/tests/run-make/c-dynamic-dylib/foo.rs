#![crate_type = "dylib"]

#[link(name = "cfoo")]
extern "C" {
    fn foo();
}

pub fn rsfoo() {
    unsafe { foo() }
}
