//@ run-pass
//@ aux-build:privacy_reexport.rs


extern crate privacy_reexport;

pub fn main() {
    // Check that public extern crates are visible to outside crates
    privacy_reexport::core::cell::Cell::new(0);

    privacy_reexport::bar::frob();
}
