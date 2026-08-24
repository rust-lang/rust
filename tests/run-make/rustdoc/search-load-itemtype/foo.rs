#![feature(extern_types, rustc_attrs, rustdoc_internals, trait_alias)]
#![allow(internal_features)]
#![no_std]

//@ has foo/keyword.while.html
//@ hasraw search.index/name/*.js while
//@ !hasraw search.index/name/*.js w_keyword
#[doc(keyword = "while")]
mod w_keyword {}

//@ has foo/primitive.u32.html
//@ hasraw search.index/name/*.js u32
//@ !hasraw search.index/name/*.js u_primitive
#[rustc_doc_primitive = "u32"]
mod u_primitive {}

//@ has foo/x_mod/index.html
//@ hasraw search.index/name/*.js x_mod
pub mod x_mod {}

//@ hasraw foo/index.html y_crate
//@ hasraw search.index/name/*.js y_crate
#[doc(no_inline)]
pub extern crate core as y_crate;

//@ hasraw foo/index.html z_import
//@ hasraw search.index/name/*.js z_import
#[doc(no_inline)]
pub use core::option as z_import;

//@ has foo/struct.AStruct.html
//@ hasraw search.index/name/*.js AStruct
pub struct AStruct {
    //@ hasraw foo/struct.AStruct.html a_structfield
    //@ hasraw search.index/name/*.js a_structfield
    pub a_structfield: i32,
}

//@ has foo/enum.AEnum.html
//@ hasraw search.index/name/*.js AEnum
pub enum AEnum {
    //@ hasraw foo/enum.AEnum.html AVariant
    //@ hasraw search.index/name/*.js AVariant
    AVariant,
}

//@ has foo/fn.a_fn.html
//@ hasraw search.index/name/*.js a_fn
pub fn a_fn() {}

//@ has foo/type.AType.html
//@ hasraw search.index/name/*.js AType
pub type AType = AStruct;

//@ has foo/static.a_static.html
//@ hasraw search.index/name/*.js a_static
pub static a_static: i32 = 1;

//@ has foo/trait.ATrait.html
//@ hasraw search.index/name/*.js ATrait
pub trait ATrait {
    //@ hasraw foo/trait.ATrait.html a_tymethod
    //@ hasraw search.index/name/*.js a_tymethod
    fn a_tymethod();
    //@ hasraw foo/trait.ATrait.html AAssocType
    //@ hasraw search.index/name/*.js AAssocType
    type AAssocType;
    //@ hasraw foo/trait.ATrait.html AAssocConst
    //@ hasraw search.index/name/*.js AAssocConst
    const AAssocConst: bool;
}

// skip ItemType::Impl, since impls are anonymous
// and have no search entry

impl AStruct {
    //@ hasraw foo/struct.AStruct.html a_method
    //@ hasraw search.index/name/*.js a_method
    pub fn a_method() {}
}

//@ has foo/macro.a_macro.html
//@ hasraw search.index/name/*.js a_macro
#[macro_export]
macro_rules! a_macro {
    () => {};
}

//@ has foo/constant.A_CONSTANT.html
//@ hasraw search.index/name/*.js A_CONSTANT
pub const A_CONSTANT: i32 = 1;

//@ has foo/union.AUnion.html
//@ hasraw search.index/name/*.js AUnion
pub union AUnion {
    //@ hasraw foo/union.AUnion.html a_unionfield
    //@ hasraw search.index/name/*.js a_unionfield
    pub a_unionfield: i32,
}

extern "C" {
    //@ has foo/foreigntype.AForeignType.html
    //@ hasraw search.index/name/*.js AForeignType
    pub type AForeignType;
}

// procattribute and procderive are defined in
// bar.rs, because they only work with proc_macro
// crate type.

//@ has foo/traitalias.ATraitAlias.html
//@ hasraw search.index/name/*.js ATraitAlias
pub trait ATraitAlias = ATrait;

//@ has foo/attribute.doc.html
//@ hasraw search.index/name/*.js doc
//@ !hasraw search.index/name/*.js aa_mod
#[doc(attribute = "doc")]
mod aa_mod {}
