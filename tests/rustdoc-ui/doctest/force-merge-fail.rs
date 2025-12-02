//@ edition: 2018
//@ compile-flags: --test --test-args=--test-threads=1 --merge-doctests=yes -Z unstable-options
//@ normalize-stderr: ".*doctest_bundle_2018.rs:\d+:\d+" -> "doctest_bundle_2018.rs:$$LINE:$$COL"

//~? ERROR failed to merge doctests

/// These two doctests will fail to force-merge, and should give a hard error as a result.
///
/// ```
/// #![deny(clashing_extern_declarations)]
/// unsafe extern "C" { fn unmangled_name() -> u8; }
/// ```
///
/// ```
/// #![deny(clashing_extern_declarations)]
/// unsafe extern "C" { fn unmangled_name(); }
/// ```
pub struct Foo;
