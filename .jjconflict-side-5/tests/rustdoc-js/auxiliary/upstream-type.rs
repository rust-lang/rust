//@ use-rustdoc-cci-doc-meta-merge

/// <https://github.com/rust-lang/rust/issues/162334>
pub struct FooBar;

/// Test case for overlapping struct and function name
#[allow(nonstandard_style)]
pub struct overlapping_name {
    _inner: (),
}

/// Test case for overlapping function and struct name
pub fn overlapping_name() -> FooBar {
    FooBar
}
