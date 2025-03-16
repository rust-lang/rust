#![no_std]

// Rustdoc JSON currently includes `#[repr(transparent)]`
// even if the transparency is not part of the public API
//
// https://doc.rust-lang.org/nomicon/other-reprs.html#reprtransparent

//@ is "$.index[*][?(@.name=='Transparent')].attrs" '["#[attr = Repr([ReprTransparent])]\n"]'
#[repr(transparent)]
pub struct Transparent(pub i64);

//@ is "$.index[*][?(@.name=='TransparentNonPub')].attrs" '["#[attr = Repr([ReprTransparent])]\n"]'
#[repr(transparent)]
pub struct TransparentNonPub(i64);

//@ is "$.index[*][?(@.name=='AllZst')].attrs" '["#[attr = Repr([ReprTransparent])]\n"]'
#[repr(transparent)]
pub struct AllZst<'a>(pub core::marker::PhantomData<&'a ()>, ());

//@ is "$.index[*][?(@.name=='AllZstNotPublic')].attrs" '["#[attr = Repr([ReprTransparent])]\n"]'
#[repr(transparent)]
pub struct AllZstNotPublic<'a>(core::marker::PhantomData<&'a ()>, ());
