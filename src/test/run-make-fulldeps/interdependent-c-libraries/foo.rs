#![crate_type = "rlib"]

#[link(name = "foo", kind = "static")]
extern {
    fn foo();
}

pub fn doit() {
    unsafe { foo(); }
}
