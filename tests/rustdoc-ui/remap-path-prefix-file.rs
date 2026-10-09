// Checks that macros uses inside doctests are correctly gated under the
// `macro` and al --remap-path-scope scopes.

//@ only-linux
//@ compile-flags: --test --test-args=--test-threads=1
//@ compile-flags: --remap-path-prefix={{src-base}}=/REMAPPED
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ rustc-env: RUST_BACKTRACE=0
//@ check-pass

//@ revisions: macro_scope diagnostics_scope documentation_scope debuginfo_scope
//@ revisions: object_scope all_scope no_scopes

//@[macro_scope] compile-flags: --remap-path-scope=macro -Z unstable-options
//@[diagnostics_scope] compile-flags: --remap-path-scope=diagnostics -Z unstable-options
//@[documentation_scope] compile-flags: --remap-path-scope=documentation -Z unstable-options
//@[debuginfo_scope] compile-flags: --remap-path-scope=debuginfo -Z unstable-options
//@[object_scope] compile-flags: --remap-path-scope=object -Z unstable-options
//@[all_scope] compile-flags: --remap-path-scope=all -Z unstable-options
// `no_scopes` passes no --remap-path-scope, so defaults to `all`

/// Only `macro`, `object` and `all` should remap what `file!()` expands to.
///
/// ```
/// #[cfg(any(macro_scope, object_scope, all_scope, no_scopes))]
/// const EXPECT_REMAPPED: bool = true;
/// #[cfg(not(any(macro_scope, object_scope, all_scope, no_scopes)))]
/// const EXPECT_REMAPPED: bool = false;
///
/// let file = file!();
/// assert_eq!(
///     file.starts_with("/REMAPPED"),
///     EXPECT_REMAPPED,
///     "unexpected file!() = {file}",
/// );
/// ```
pub fn f() {}
