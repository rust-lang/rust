#![deny(rustdoc::broken_intra_doc_links)]
#![feature(lang_items)]
#![feature(no_core)]
#![feature(rustdoc_internals)]
#![no_core]

#[lang = "usize"]
/// [Self::f]
/// [Self::MAX]
// @has intra_link_prim_self/primitive.usize.html
// @has - '//a[@href="primitive.usize.html#method.f"]' 'Self::f'
// @has - '//a[@href="primitive.usize.html#associatedconstant.MAX"]' 'Self::MAX'
impl usize {
    /// Some docs
    pub fn f() {}

    /// 10 and 2^32 are basically the same.
    pub const MAX: usize = 10;

    // FIXME(#8995) uncomment this when associated types in inherent impls are supported
    // @ has - '//a[@href="{{channel}}/std/primitive.usize.html#associatedtype.ME"]' 'Self::ME'
    // / [Self::ME]
    //pub type ME = usize;
}

#[doc(primitive = "usize")]
/// This has some docs.
mod usize {}

/// [S::f]
/// [Self::f]
pub struct S;

impl S {
    pub fn f() {}
}
