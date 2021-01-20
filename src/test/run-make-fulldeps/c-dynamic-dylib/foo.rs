#![crate_type = "dylib"]

#[link(name = "cfoo")]
extern {
    fn foo();
}

pub fn rsfoo() {
    unsafe { foo() }
}
