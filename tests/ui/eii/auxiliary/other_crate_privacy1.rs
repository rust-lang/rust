//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
fn decl1(x: u64);

mod private {
    #[eii(eii2)]
    pub fn decl2(x: u64);
}

pub use private::eii2 as eii3;
pub use private::decl2 as decl3;

pub fn local_call_decl1(x: u64) {
    decl1(x)
}
