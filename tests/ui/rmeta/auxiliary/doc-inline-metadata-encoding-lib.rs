#![crate_type = "lib"]
#![crate_name = "doc_inline_metadata_encoding_lib"]

mod inner {
    pub struct PublicItem {
        pub field: i32,
    }

    pub struct InnerItem {
        pub field: i32,
    }

    pub struct MixedAttributes {
        pub field: i32,
    }
}

#[doc(inline)]
pub use inner::PublicItem;

#[doc(inline)]
pub use inner::InnerItem;

// Regression test for #149919.
#[doc(hidden)]
#[doc(inline)]
pub use inner::MixedAttributes;
