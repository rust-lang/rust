#![feature(linkage)]
#![crate_type = "lib"]

extern "C" {
    #[linkage = "external"]
    pub static collision: *const i32;
}
