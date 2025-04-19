//! Ensure that `for_each_function!` isn't missing any symbols.

use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::process::Command;

macro_rules! callback {
    (
        fn_name: $name:ident,
        attrs: [$($attr:meta),*],
        extra: [$set:ident],
    ) => {
        let name = stringify!($name);
        let new = $set.insert(name);
        assert!(new, "duplicate function `{name}` in `ALL_OPERATIONS`");
    };
}

#[test]
fn test_for_each_function_all_included() {
    let all_functions: HashSet<_> = include_str!("../../../etc/function-list.txt")
        .lines()
        .filter(|line| !line.starts_with("#"))
        .collect();

    let mut tested = HashSet::new();

    libm_macros::for_each_function! {
        callback: callback,
        extra: [tested],
    };

    let untested = all_functions.difference(&tested);
    if untested.clone().next().is_some() {
        panic!(
            "missing tests for the following: {untested:#?} \
            \nmake sure any new functions are entered in \
            `ALL_OPERATIONS` (in `libm-macros`)."
        );
    }
    assert_eq!(all_functions, tested);
}

#[test]
fn ensure_list_updated() {
    if libm_test::ci() {
        // Most CI tests run in Docker where we don't have Python or Rustdoc, so it's easiest
        // to just run the python file directly when it is available.
        eprintln!("skipping test; CI runs the python file directly");
        return;
    }

    let res = Command::new("python3")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/update-api-list.py"))
        .arg("--check")
        .status()
        .unwrap();

    assert!(res.success(), "May need to run `./etc/update-api-list.py`");
}
