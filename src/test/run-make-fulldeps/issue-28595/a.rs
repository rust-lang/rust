#![crate_type = "rlib"]

#[link(name = "a", kind = "static")]
extern {
    pub fn a();
}
