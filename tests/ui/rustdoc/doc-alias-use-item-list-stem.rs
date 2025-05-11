// Check that we don't ICE on `#[doc(alias)]`es placed on use items with list stems.
// issue: <https://github.com/rust-lang/rust/issues/138723>
//@ check-pass

#[doc(alias = "empty")]
pub use {};

#[doc(alias = "id")]
pub use {std::convert::identity};

fn main() {}
