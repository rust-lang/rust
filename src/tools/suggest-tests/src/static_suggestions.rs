use std::sync::OnceLock;

use crate::{Suggestion, sug};

// FIXME: perhaps this could use `std::lazy` when it is stablizied
macro_rules! static_suggestions {
    ($( [ $( $glob:expr ),* $(,)? ] => [ $( $suggestion:expr ),* $(,)? ] ),* $(,)? ) => {
        pub(crate) fn static_suggestions() -> &'static [(Vec<&'static str>, Vec<Suggestion>)]
        {
            static S: OnceLock<Vec<(Vec<&'static str>, Vec<Suggestion>)>> = OnceLock::new();
            S.get_or_init(|| vec![ $( (vec![ $($glob),* ], vec![ $($suggestion),* ]) ),*])
        }
    }
}

static_suggestions! {
    ["*.md"] => [
        sug!("test", 0, ["linkchecker"]),
    ],

    ["compiler/*"] => [
        sug!("check"),
        sug!("test", 1, ["tests/ui", "tests/run-make"]),
    ],

    ["compiler/rustc_mir_transform/*"] => [
        sug!("test", 1, ["mir-opt"]),
    ],

    [
        "compiler/rustc_mir_transform/src/coverage/*",
        "compiler/rustc_codegen_llvm/src/coverageinfo/*",
    ] => [
        sug!("test", 1, ["coverage"]),
    ],

    ["src/librustdoc/*"] => [
        sug!("test", 1, ["rustdoc"]),
    ],
}
