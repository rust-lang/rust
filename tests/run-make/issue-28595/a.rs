#![crate_type = "rlib"]

#[link(name = "a", kind = "static")]
extern "C" {
    pub fn a();
}
