#![crate_type = "rlib"]

#[link(name = "cfoo")]
extern {
    fn foo();
}

pub fn rsfoo() {
    unsafe { foo() }
}
