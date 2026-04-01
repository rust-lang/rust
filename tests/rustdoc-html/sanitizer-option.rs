//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ compile-flags: --test -Z sanitizer=address -C unsafe-allow-abi-mismatch=sanitizer
//
// #43031: Verify that rustdoc passes `-Z` options to rustc. Use an extern
// function that is provided by the sanitizer runtime, if flag is not passed
// correctly, then linking will fail.

/// ```
/// extern "C" {
///     fn __sanitizer_print_stack_trace();
/// }
///
/// fn main() {
///     unsafe { __sanitizer_print_stack_trace() };
/// }
/// ```
pub fn z_flag_is_passed_to_rustc() {}
