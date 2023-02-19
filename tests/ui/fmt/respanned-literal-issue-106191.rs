// aux-build:format-string-proc-macro.rs
// check-fail
// known-bug: #106191
// unset-rustc-env:RUST_BACKTRACE
// had to be reverted
// error-pattern:unexpectedly panicked
// failure-status:101
// dont-check-compiler-stderr

extern crate format_string_proc_macro;

fn main() {
    format_string_proc_macro::respan_to_invalid_format_literal!("¡");
    format_args!(r#concat!("¡        {"));
}
