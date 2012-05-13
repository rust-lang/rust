// Define the rustc API's that the metadata module has access to
// Over time we will reduce these dependencies and, once metadata has
// no dependencies on rustc it can move into its own crate.

mod middle {
    import ast_map = middle_::ast_map;
    export ast_map;
    import ty = middle_::ty;
    export ty;
    import trans = middle_::trans;
    export trans;
    import typeck = middle_::typeck;
    export typeck;
    import last_use = middle_::last_use;
    export last_use;
    import freevars = middle_::freevars;
    export freevars;
    import resolve = middle_::resolve;
    export resolve;
}

mod front {
}

mod back {
}

mod driver {
    import session = driver_::session;
    export session;
    import diagnostic = rustsyntax::diagnostic;
    export diagnostic;
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
