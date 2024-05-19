#![feature(extern_types)]

extern "C" {
    pub type CrossCrate;
}

impl CrossCrate {
    pub fn foo(&self) {}
}
