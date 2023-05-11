#![feature(rustc_attrs)]

#[rustc_has_incoherent_inherent_impls]
pub struct StructWithAttr;
pub struct StructNoAttr;

#[rustc_has_incoherent_inherent_impls]
pub enum EnumWithAttr {}
pub enum EnumNoAttr {}
