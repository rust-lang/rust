#![crate_type = "dylib"]

#[link(name = "foo", kind = "static")]
extern "C" {
    pub fn foo();
}
