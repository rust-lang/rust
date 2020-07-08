// compile-flags: --document-private-items
#![deny(intra_doc_link_resolution_failure)]

// Linking from a private item to a private type is fine with --document-private-items.

struct Private;

pub struct Public {
    /// [`Private`]
    private: Private,
}
