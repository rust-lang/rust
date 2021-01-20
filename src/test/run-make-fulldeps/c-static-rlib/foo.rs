#![crate_type = "rlib"]

#[link(name = "cfoo", kind = "static")]
extern {
    fn foo();
}

pub fn rsfoo() {
    unsafe { foo() }
}
