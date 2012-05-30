// Define the rustc API's that the metadata module has access to
// Over time we will reduce these dependencies and, once metadata has
// no dependencies on rustc it can move into its own crate.

mod middle {
    import ty = middle_::ty;
    export ty;
}

mod front {
}

mod back {
}

mod driver {
}

mod util {
    import ppaux = util_::ppaux;
    export ppaux;
}

mod lib {
    import llvm = lib_::llvm;
    export llvm;
}
