//@ aux-crate:macro_provenance_leaf=macro_provenance_leaf.rs

extern crate macro_provenance_leaf;

pub use macro_provenance_leaf::{Hidden, captured_type, definition_side, nested};

#[macro_export]
macro_rules! call_site_private {
    ($name:ident) => {
        pub fn $name() -> private_macros::Hidden {
            loop {}
        }
    };
}
