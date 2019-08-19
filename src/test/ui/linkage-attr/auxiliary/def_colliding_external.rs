#![feature(linkage)]
#![crate_type = "lib"]

extern {
    #[linkage="external"]
    pub static collision: *const i32;
}
