//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

// does have an impl but can't be called
#[eii(eii1)]
fn decl1(x: u64);

#[eii(eii2)]
pub fn decl2(x: u64);

mod private {
    #[eii(eii3)]
    pub fn decl3(x: u64);
}

pub use private::eii3 as eii4;
pub use private::decl3 as decl4;

pub fn local_call_decl1(x: u64) {
    decl1(x)
}
