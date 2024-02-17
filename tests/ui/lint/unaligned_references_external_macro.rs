//@ aux-build:unaligned_references_external_crate.rs

extern crate unaligned_references_external_crate;

unaligned_references_external_crate::mac! { //~ERROR reference to packed field is unaligned
    #[repr(packed)]
    pub struct X {
        pub field: u16
    }
}

fn main() {}
