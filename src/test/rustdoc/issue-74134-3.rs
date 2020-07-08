// should-fail
#![deny(intra_doc_link_resolution_failure)]

// Linking from a public item to a private type fails without --document-private-items.

struct Private;

pub struct Public {
    /// [`Private`]
    pub public: u32,
}
