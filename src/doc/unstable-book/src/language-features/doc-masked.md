# `doc_masked`

The tracking issue for this feature is: [TODO](TODO)

-----

The `doc_masked` feature allows a crate to exclude types from a given crate from appearing in lists
of trait implementations. The specifics of the feature are as follows:

1. When rustdoc encounters an `extern crate` statement annotated with a `#[doc(masked)]` attribute,
   it marks the crate as being masked.

2. When listing traits a given type implements, rustdoc ensures that traits from masked crates are
   not emitted into the documentation.

3. When listing types that implement a given trait, rustdoc ensures that types from masked crates
   are not emitted into the documentation.

This feature was introduced in PR [TODO](TODO) to ensure that compiler-internal and
implementation-specific types and traits were not included in the standard library's documentation.
Such types would introduce broken links into the documentation.

```rust
#![feature(doc_masked)]

// Since std is automatically imported, we need to import it into a separate name to apply the
// attribute. This is used as a simple demonstration, but any extern crate statement will suffice.
#[doc(masked)]
extern crate std as realstd;

/// A sample marker trait that exists on floating-point numbers, even though this page won't list
/// them!
pub trait MyMarker { }

impl MyMarker for f32 { }
impl MyMarker for f64 { }
```
