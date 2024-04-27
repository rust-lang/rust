#![crate_type = "lib"]

pub struct Struct(pub u32);

impl Drop for Struct {
    fn drop(&mut self) {}
}
