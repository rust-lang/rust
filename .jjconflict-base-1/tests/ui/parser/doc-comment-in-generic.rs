//! Tests correct parsing of doc comments on generic parameters in traits.
//! Checks that compiler doesn't panic when processing this.

//@ check-pass

#![crate_type = "lib"]

pub trait Layer<
    /// Documentation for generic parameter.
    Input,
>
{
}
