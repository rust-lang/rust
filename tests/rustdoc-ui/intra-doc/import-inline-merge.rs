// Import for `A` is inlined and doc comments on the import and `A` itself are merged.
// After the merge they still have correct parent scopes to resolve both `[A]` and `[B]`.

// check-pass

#![allow(rustdoc::private_intra_doc_links)]

mod m {
    /// [B]
    pub struct A {}

    pub struct B {}
}

/// [A]
pub use m::A;
