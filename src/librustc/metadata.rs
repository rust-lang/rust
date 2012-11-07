// Define the rustc API's that the metadata module has access to
// Over time we will reduce these dependencies and, once metadata has
// no dependencies on rustc it can move into its own crate.

mod middle {
    #[legacy_exports];
    pub use middle_::ty;
    pub use middle_::resolve;
}

mod front {
    #[legacy_exports];
}

mod back {
    #[legacy_exports];
}

mod driver {
    #[legacy_exports];
}

mod util {
    #[legacy_exports];
    pub use util_::ppaux;
}

mod lib {
    #[legacy_exports];
    pub use lib_::llvm;
}
