// check-pass
// Check that it doesn't panic when `Input` gets its visibility checked.

#![crate_type = "lib"]

pub trait Layer<
    /// Hello.
    Input,
> {}
