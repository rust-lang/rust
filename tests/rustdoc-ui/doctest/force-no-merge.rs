//@ edition: 2024
//@ check-pass
//@ compile-flags: --test --test-args=--test-threads=1 --merge-doctests=no -Z unstable-options
//@ normalize-stderr: ".*doctest_bundle_2018.rs:\d+:\d+" -> "doctest_bundle_2018.rs:$$LINE:$$COL"

//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ normalize-stdout: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

/// These two doctests should not force-merge, even though this crate has edition 2024 and the
/// individual doctests are not annotated.
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
