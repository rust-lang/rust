// Define the rustc API's that the metadata module has access to
// Over time we will reduce these dependencies and, once metadata has
// no dependencies on rustc it can move into its own crate.

mod middle {
    pub use middle_::ty;
    pub use middle_::resolve;
}

mod front {
}

mod back {
}

mod driver {
}

mod util {
    pub use util_::ppaux;
}

mod lib {
    pub use lib_::llvm;
}
