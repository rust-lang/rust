#![crate_type = "rlib"]

#[link(name = "thin_archive", kind = "static")]
extern "C" {
    pub fn simple_fn();
}
