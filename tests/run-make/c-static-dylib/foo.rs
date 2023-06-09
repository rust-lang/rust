#![crate_type = "dylib"]

#[link(name = "cfoo", kind = "static")]
extern "C" {
    fn foo();
}

pub fn rsfoo() {
    unsafe { foo() }
}
