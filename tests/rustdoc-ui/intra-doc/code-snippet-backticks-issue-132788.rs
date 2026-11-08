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
//~| HELP if you intended to link to external documentation
pub struct OrdinaryBrokenLink;


/// Access arr[0].
//~^ ERROR unresolved link to `0`
//~| HELP try marking as source code
pub struct NumericIndexing;


/// Access state.arr[idx].
//~^ ERROR unresolved link to `idx`
//~| HELP try marking as source code
pub struct FieldIndexing;

/// Access module::ARR[idx].
//~^ ERROR unresolved link to `idx`
//~| HELP try marking as source code
pub struct QualifiedPathIndexing;

// Malformed receiver syntax should retain the existing fallback advice.
/// Access state:::arr[idx].
//~^ ERROR unresolved link to `idx`
//~| HELP to escape `[` and `]` characters
//~| HELP if you intended to link to external documentation
pub struct MalformedReceiver;
