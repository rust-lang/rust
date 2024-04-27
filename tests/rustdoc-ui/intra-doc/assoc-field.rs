// Traits in scope are collected for doc links in field attributes.

//@ check-pass
//@ aux-build: assoc-field-dep.rs

extern crate assoc_field_dep;
pub use assoc_field_dep::*;

#[derive(Clone)]
pub struct Struct;

pub mod mod1 {
    pub struct Fields {
        /// [crate::Struct::clone]
        pub field: u8,
    }
}

pub mod mod2 {
    pub enum Fields {
        V {
            /// [crate::Struct::clone]
            field: u8,
        },
    }
}
