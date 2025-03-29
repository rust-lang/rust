#![deny(unused_attributes)]
// Unused attributes on macro_rules requires special handling since the
// macro_rules definition does not survive towards HIR.

// A sample of various built-in attributes.
#[macro_export]
#[macro_use] //~ ERROR `#[macro_use]` only has an effect
#[path="foo"] //~ ERROR #[path]` only has an effect
#[recursion_limit="1"] //~ ERROR crate-level attribute should be an inner attribute
macro_rules! foo {
    () => {};
}

// The following should not warn about unused attributes.
#[allow(unused)]
macro_rules! foo2 {
    () => {};
}

#[cfg(false)]
macro_rules! foo {
    () => {};
}

/// Some docs
#[deprecated]
#[doc = "more docs"]
#[macro_export]
macro_rules! bar {
    () => {};
}

fn main() {}
