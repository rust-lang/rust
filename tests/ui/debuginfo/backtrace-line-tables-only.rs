// Test that when debug info only includes line tables that backtrace is still generated
// successfully. This previously failed when compiling with `clang -g1`.
// Part of <https://github.com/rust-lang/rust/issues/122899> porting some backtrace tests to rustc.
// ignore-tidy-linelength
//@ ignore-windows original test is ignore-windows
//@ ignore-android FIXME #17520
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-fuchsia Backtraces not symbolized
//@ needs-unwind
//@ run-pass
//@ compile-flags: -Cdebuginfo=line-tables-only -Cstrip=none
#![feature(backtrace_frames)]

use std::backtrace::{self, Backtrace};
use std::ffi::c_void;
use std::ptr::addr_of_mut;

pub type Callback = extern "C" fn(data: *mut c_void);

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn line_tables_only_foo(cb: Callback, data: *mut c_void);
}

extern "C" fn store_backtrace(data: *mut c_void) {
    let bt = backtrace::Backtrace::capture();
    unsafe { *data.cast::<Option<Backtrace>>() = Some(bt) };
}

fn assert_contains(
    backtrace: &Backtrace,
    expected_name: &str,
    expected_file: &str,
    expected_line: u32,
) {
    // FIXME(jieyouxu): fix this ugly fragile test when `BacktraceFrame` has accessors like...
    // `symbols()`.
    let backtrace = format!("{:#?}", backtrace);
    eprintln!("{}", backtrace);
    assert!(backtrace.contains(expected_name), "backtrace does not contain expected name {}", expected_name);
    assert!(backtrace.contains(expected_file), "backtrace does not contain expected file {}", expected_file);
    assert!(backtrace.contains(&expected_line.to_string()), "backtrace does not contain expected line {}", expected_line);
}

/// Verifies that when debug info includes only lines tables the generated
/// backtrace is still generated successfully. The test exercises behaviour
/// that failed previously when compiling with clang -g1.
///
/// The test case uses C rather than rust, since at that time when it was
/// written the debug info generated at level 1 in rustc was essentially
/// the same as at level 2.
fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let mut backtrace: Option<Backtrace> = None;
    unsafe { line_tables_only_foo(store_backtrace, addr_of_mut!(backtrace).cast::<c_void>()) };
    let backtrace = backtrace.expect("backtrace");
    assert_contains(&backtrace, "line_tables_only_foo", "rust_test_helpers.c", 435);
    assert_contains(&backtrace, "line_tables_only_bar", "rust_test_helpers.c", 439);
    assert_contains(&backtrace, "line_tables_only_baz", "rust_test_helpers.c", 443);
}
