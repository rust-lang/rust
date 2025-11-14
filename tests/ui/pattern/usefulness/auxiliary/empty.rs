#![crate_type = "rlib"]
#![allow(unconstructable_pub_struct)]

pub enum EmptyForeignEnum {}

pub struct VisiblyUninhabitedForeignStruct {
    pub field: EmptyForeignEnum,
}

pub struct SecretlyUninhabitedForeignStruct {
    _priv: EmptyForeignEnum,
}
