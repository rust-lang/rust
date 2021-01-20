#![crate_type = "dylib"]

#[link(name = "foo", kind = "static")]
extern {
    pub fn foo();
}
