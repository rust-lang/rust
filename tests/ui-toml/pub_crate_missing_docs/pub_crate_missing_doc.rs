//! this is crate
#![warn(clippy::missing_docs_in_private_items)]

/// this is mod
mod my_mod {
    /// some docs
    fn priv_with_docs() {}
    fn priv_no_docs() {}
    /// some docs
    pub(crate) fn crate_with_docs() {}
    pub(crate) fn crate_no_docs() {}
    /// some docs
    pub(super) fn super_with_docs() {}
    pub(super) fn super_no_docs() {}

    mod my_sub {
        /// some docs
        fn sub_priv_with_docs() {}
        fn sub_priv_no_docs() {}
        /// some docs
        pub(crate) fn sub_crate_with_docs() {}
        pub(crate) fn sub_crate_no_docs() {}
        /// some docs
        pub(super) fn sub_super_with_docs() {}
        pub(super) fn sub_super_no_docs() {}
    }
}

fn main() {
    my_mod::crate_with_docs();
    my_mod::crate_no_docs();
}
