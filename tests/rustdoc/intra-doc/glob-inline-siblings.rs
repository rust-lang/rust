#![deny(rustdoc::broken_intra_doc_links)]
#![allow(nonstandard_style)]

// Reduced case from `core::arch::{arm, aarch64}`.
// Both versions of arm should link to the struct that they inlined.
// aarch64 should not link to the inlined copy in arm 32.

mod arm_shared {
    pub struct uint32x4_t;
    pub struct uint32x4x2_t(pub uint32x4_t, pub uint32x4_t);
    pub fn vabaq_u32(
        _a: uint32x4_t,
        _b: uint32x4_t,
        _c: uint32x4_t
    ) -> uint32x4_t {
        uint32x4_t
    }
}

pub mod arm {
    pub use crate::arm_shared::*;
    // @has glob_inline_siblings/arm/fn.vabaq_u32.html '//a[@href="struct.uint32x4_t.html"]' 'uint32x4_t'
    // @has glob_inline_siblings/arm/struct.uint32x4x2_t.html '//a[@href="struct.uint32x4_t.html"]' 'uint32x4_t'
    // @!has glob_inline_siblings/arm/fn.vabaq_u32.html '//a[@href="../aarch64/struct.uint32x4_t.html"]' 'uint32x4_t'
    // @!has glob_inline_siblings/arm/struct.uint32x4x2_t.html '//a[@href="../aarch64/struct.uint32x4_t.html"]' 'uint32x4_t'

    /// [uint32x4_t]
    // @has glob_inline_siblings/arm/struct.LocalThing.html '//a[@href="struct.uint32x4_t.html"]' 'uint32x4_t'
    // @!has glob_inline_siblings/arm/struct.LocalThing.html '//a[@href="../aarch64/struct.uint32x4_t.html"]' 'uint32x4_t'
    pub struct LocalThing;
}


pub mod aarch64 {
    pub use crate::arm_shared::*;
    // @has glob_inline_siblings/aarch64/fn.vabaq_u32.html '//a[@href="struct.uint32x4_t.html"]' 'uint32x4_t'
    // @has glob_inline_siblings/aarch64/struct.uint32x4x2_t.html '//a[@href="struct.uint32x4_t.html"]' 'uint32x4_t'
    // @!has glob_inline_siblings/aarch64/fn.vabaq_u32.html '//a[@href="../arm/struct.uint32x4_t.html"]' 'uint32x4_t'
    // @!has glob_inline_siblings/aarch64/struct.uint32x4x2_t.html '//a[@href="../arm/struct.uint32x4_t.html"]' 'uint32x4_t'

    /// [uint32x4_t]
    // @has glob_inline_siblings/aarch64/struct.LocalThing.html '//a[@href="struct.uint32x4_t.html"]' 'uint32x4_t'
    // @!has glob_inline_siblings/aarch64/struct.LocalThing.html '//a[@href="../arm/struct.uint32x4_t.html"]' 'uint32x4_t'
    pub struct LocalThing;
}
