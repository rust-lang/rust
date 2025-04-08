//@ edition: 2024
//@ run-pass
//@ rustc-env: MY_RUSTC_ENV = my-rustc-value
//@ exec-env: MY_EXEC_ENV = my-exec-value

// Check that compiletest trims whitespace from environment variable names
// specified in `rustc-env` and `exec-env` directives, so that
// `//@ exec-env: FOO=bar` sees the name as `FOO` and not ` FOO`.
//
// Values are currently not trimmed.

fn main() {
    assert_eq!(option_env!("MY_RUSTC_ENV"), Some(" my-rustc-value"));
    assert_eq!(std::env::var("MY_EXEC_ENV").as_deref(), Ok(" my-exec-value"));
}
