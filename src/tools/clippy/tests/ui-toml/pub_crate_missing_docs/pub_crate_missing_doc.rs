//! this is crate
#![allow(missing_docs)]
#![allow(clippy::struct_field_names)]
#![warn(clippy::missing_docs_in_private_items)]

/// this is mod
mod my_mod {
    /// some docs
    fn priv_with_docs() {}
    fn priv_no_docs() {}
    /// some docs
    pub(crate) fn crate_with_docs() {}
    pub(crate) fn crate_no_docs() {}
    //~^ missing_docs_in_private_items
    /// some docs
    pub(super) fn super_with_docs() {}
    pub(super) fn super_no_docs() {}
    //~^ missing_docs_in_private_items

    mod my_sub {
        /// some docs
        fn sub_priv_with_docs() {}
        fn sub_priv_no_docs() {}
        /// some docs
        pub(crate) fn sub_crate_with_docs() {}
        pub(crate) fn sub_crate_no_docs() {}
        //~^ missing_docs_in_private_items
        /// some docs
        pub(super) fn sub_super_with_docs() {}
        pub(super) fn sub_super_no_docs() {}
    }

    /// some docs
    pub(crate) struct CrateStructWithDocs {
        /// some docs
        pub(crate) crate_field_with_docs: (),
        pub(crate) crate_field_no_docs: (),
        //~^ missing_docs_in_private_items
        /// some docs
        priv_field_with_docs: (),
        priv_field_no_docs: (),
    }

    pub(crate) struct CrateStructNoDocs {
        //~^ missing_docs_in_private_items
        /// some docs
        pub(crate) crate_field_with_docs: (),
        pub(crate) crate_field_no_docs: (),
        //~^ missing_docs_in_private_items
        /// some docs
        priv_field_with_docs: (),
        priv_field_no_docs: (),
    }
}

/// some docs
type CrateTypedefWithDocs = String;
type CrateTypedefNoDocs = String;
//~^ missing_docs_in_private_items
/// some docs
pub type PubTypedefWithDocs = String;
pub type PubTypedefNoDocs = String;

fn main() {
    my_mod::crate_with_docs();
    my_mod::crate_no_docs();
}
