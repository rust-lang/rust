// https://github.com/rust-lang/rust/issues/132788
#![deny(rustdoc::broken_intra_doc_links)]

/// Access arr[idx].
//~^ ERROR unresolved link to `idx`
//~| HELP try marking as source
pub struct UnformattedIndexing;

/// Access `arr[idx]`.
pub struct AlreadyFormattedIndexing;

/// See [DefinitelyMissingType].
//~^ ERROR unresolved link to `DefinitelyMissingType`
//~| HELP to escape `[` and `]` characters
pub struct OrdinaryBrokenLink;
