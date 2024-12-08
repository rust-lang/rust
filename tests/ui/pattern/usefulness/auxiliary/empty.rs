#![crate_type = "rlib"]
pub enum EmptyForeignEnum {}

pub struct VisiblyUninhabitedForeignStruct {
    pub field: EmptyForeignEnum,
}

pub struct SecretlyUninhabitedForeignStruct {
    _priv: EmptyForeignEnum,
}
