use super::argument::ArgumentList;
use super::intrinsic_helpers::{IntrinsicTypeDefinition, TypeKind};

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic<T: IntrinsicTypeDefinition> {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsic.
    pub arguments: ArgumentList<T>,

    /// The return type of this intrinsic.
    pub results: T,

    /// Any architecture-specific tags.
    pub arch_tags: Vec<String>,
}

pub fn format_f16_return_value<T: IntrinsicTypeDefinition>(intrinsic: &Intrinsic<T>) -> String {
    // the `intrinsic-test` crate compares the output of C and Rust intrinsics. Currently, It uses
    // a string representation of the output value to compare. In C, f16 values are currently printed
    // as hexadecimal integers. Since https://github.com/rust-lang/rust/pull/127013, rust does print
    // them as decimal floating point values. To keep the intrinsics tests working, for now, format
    // vectors containing f16 values like C prints them.
    let return_value = match intrinsic.results.kind() {
        TypeKind::Float if intrinsic.results.inner_size() == 16 => "debug_f16(__return_value)",
        _ => "format_args!(\"{__return_value:.150?}\")",
    };

    String::from(return_value)
}
