#![no_std]

// Rustdoc JSON *only* includes `#[repr(transparent)]`
// if the transparency is public API:
// - if a non-1-ZST field exists, it has to be public
// - otherwise, all fields are 1-ZST and at least one of them is public
//
// More info: https://doc.rust-lang.org/nomicon/other-reprs.html#reprtransparent

// Here, the non-1-ZST field is public.
// We expect `#[repr(transparent)]` in the attributes.
//
//@ is "$.index[?(@.name=='Transparent')].attrs" '["#[repr(transparent)]"]'
#[repr(transparent)]
pub struct Transparent(pub i64);

// Here the non-1-ZST field isn't public, so the attribute isn't included.
//
//@ has "$.index[?(@.name=='TransparentNonPub')]"
//@ is "$.index[?(@.name=='TransparentNonPub')].attrs" '[]'
#[repr(transparent)]
pub struct TransparentNonPub(i64);

// Only 1-ZST fields here, and one of them is public.
// We expect `#[repr(transparent)]` in the attributes.
//
//@ is "$.index[?(@.name=='AllZst')].attrs" '["#[repr(transparent)]"]'
#[repr(transparent)]
pub struct AllZst<'a>(pub core::marker::PhantomData<&'a ()>, ());

// Only 1-ZST fields here but none of them are public.
// The attribute isn't included.
//
//@ has "$.index[?(@.name=='AllZstNotPublic')]"
//@ is "$.index[?(@.name=='AllZstNotPublic')].attrs" '[]'
#[repr(transparent)]
pub struct AllZstNotPublic<'a>(core::marker::PhantomData<&'a ()>, ());
