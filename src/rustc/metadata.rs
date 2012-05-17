// Define the rustc API's that the metadata module has access to
// Over time we will reduce these dependencies and, once metadata has
// no dependencies on rustc it can move into its own crate.

mod middle {
    import ast_map = middle_::ast_map;
    export ast_map;
    import ty = middle_::ty;
    export ty;
}

mod front {
}

mod back {
}

mod driver {
    import session = driver_::session;
    export session;
}

mod util {
    import common = util_::common;
    export common;
    import ppaux = util_::ppaux;
    export ppaux;
    import filesearch = util_::filesearch;
    export filesearch;
}

mod lib {
    import llvm = lib_::llvm;
    export llvm;
}
