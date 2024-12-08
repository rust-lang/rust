//@ revisions: public private
//@ [private]compile-flags: --document-private-items
//@ check-pass

// There are 4 cases here:
// 1. public item  -> public type:  no warning
// 2. public item  -> private type: warning
// 3. private item -> public type:  no warning
// 4. private item -> private type: no warning
// All 4 cases are tested with and without --document-private-items.
//
// Case 4 without --document-private-items is the one described in issue #74134.

struct PrivateType;
pub struct PublicType;

pub struct Public {
    /// [`PublicType`]
    /// [`PrivateType`]
    //~^ WARNING public documentation for `public_item` links to private item `PrivateType`
    pub public_item: u32,

    /// [`PublicType`]
    /// [`PrivateType`]
    private_item: u32,
}

// The following cases are identical to the ones above, except that they are in a private
// module. Thus they all fall into cases 3 and 4 and should not produce a warning.

mod private {
    pub struct Public {
        /// [`super::PublicType`]
        /// [`super::PrivateType`]
        pub public_item: u32,

        /// [`super::PublicType`]
        /// [`super::PrivateType`]
        private_item: u32,
    }
}
