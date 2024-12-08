# `doc_masked`

The tracking issue for this feature is: [#44027]

-----

The `doc_masked` feature allows a crate to exclude types from a given crate from appearing in lists
of trait implementations. The specifics of the feature are as follows:

1. When rustdoc encounters an `extern crate` statement annotated with a `#[doc(masked)]` attribute,
   it marks the crate as being masked.

2. When listing traits a given type implements, rustdoc ensures that traits from masked crates are
   not emitted into the documentation.

3. When listing types that implement a given trait, rustdoc ensures that types from masked crates
   are not emitted into the documentation.

This feature was introduced in PR [#44026] to ensure that compiler-internal and
implementation-specific types and traits were not included in the standard library's documentation.
Such types would introduce broken links into the documentation.

[#44026]: https://github.com/rust-lang/rust/pull/44026
[#44027]: https://github.com/rust-lang/rust/pull/44027
