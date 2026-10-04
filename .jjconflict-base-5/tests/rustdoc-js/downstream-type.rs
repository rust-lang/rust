//@ aux-crate:upstream_type=upstream-type.rs
//@ aux-build:upstream-type.rs
//@ build-aux-docs
//@ use-rustdoc-cci-doc-meta-merge

/// <https://github.com/rust-lang/rust/issues/162334>
pub fn downstream_fn(f: upstream_type::FooBar) {}

pub fn with_overlap(f: upstream_type::overlapping_name) {}
