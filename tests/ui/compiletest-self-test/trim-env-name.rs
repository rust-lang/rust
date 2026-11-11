//@ edition: 2024
//@ revisions: set unset
//@ run-pass
//@ ignore-cross-compile (assume that non-cross targets have working env vars)
//@ rustc-env: MY_RUSTC_ENV =  my-rustc-value
//@ exec-env:  MY_EXEC_ENV  =  my-exec-value
//@[unset] unset-rustc-env:    MY_RUSTC_ENV
//@[unset] unset-exec-env:     MY_EXEC_ENV

// Check that compiletest trims whitespace from environment variable names
// specified in `rustc-env` and `exec-env` directives, so that
// `//@ exec-env: FOO=bar` sees the name as `FOO` and not ` FOO`.
//
// Values are currently not trimmed.
//
// Since this is a compiletest self-test, only run it on non-cross targets,
// to avoid having to worry about weird targets that don't support env vars.

fn main() {
    let is_set = cfg!(set);
    assert_eq!(option_env!("MY_RUSTC_ENV"), is_set.then_some("  my-rustc-value"));
    assert_eq!(std::env::var("MY_EXEC_ENV").ok().as_deref(), is_set.then_some("  my-exec-value"));
}
