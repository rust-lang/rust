pub mod longhands {
    pub use super::*;

    pub use super::common_types::computed::compute_CSSColor as to_computed_value;

    pub fn computed_as_specified() {}
}

pub mod common_types {
    pub mod computed {
        pub use super::super::longhands::computed_as_specified as compute_CSSColor;
    }
}
