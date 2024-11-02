//! Ensure that `for_each_function!` isn't missing any symbols.

/// Files in `src/` that do not export a testable symbol.
const ALLOWED_SKIPS: &[&str] = &[
    // Not a generic test function
    "fenv",
    // Nonpublic functions
    "expo2",
    "k_cos",
    "k_cosf",
    "k_expo2",
    "k_expo2f",
    "k_sin",
    "k_sinf",
    "k_tan",
    "k_tanf",
    "rem_pio2",
    "rem_pio2_large",
    "rem_pio2f",
];

macro_rules! callback {
    (
        fn_name: $name:ident,
        extra: [$push_to:ident],
    ) => {
        $push_to.push(stringify!($name));
    };
}

#[test]
fn test_for_each_function_all_included() {
    let mut included = Vec::new();
    let mut missing = Vec::new();

    libm_macros::for_each_function! {
        callback: callback,
        extra: [included],
    };

    for f in libm_test::ALL_FUNCTIONS {
        if !included.contains(f) && !ALLOWED_SKIPS.contains(f) {
            missing.push(f)
        }
    }

    if !missing.is_empty() {
        panic!(
            "missing tests for the following: {missing:#?} \
            \nmake sure any new functions are entered in \
            `ALL_FUNCTIONS` (in `libm-macros`)."
        );
    }
}
