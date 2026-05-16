#![crate_type = "rlib"]
#![allow(unused_unconstructable_pub_structs)]
pub enum EmptyForeignEnum {}

pub struct VisiblyUninhabitedForeignStruct {
    pub field: EmptyForeignEnum,
}

pub struct SecretlyUninhabitedForeignStruct {
    _priv: EmptyForeignEnum,
}
