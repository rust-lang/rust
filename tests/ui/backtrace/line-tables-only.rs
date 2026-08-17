// ignore-tidy-file-linelength
//
// Test that when debug info only includes line tables that backtrace is still generated
// successfully.
// Original test:
// <https://github.com/rust-lang/backtrace-rs/tree/6fa4b85b9962c3e1be8c2e5cc605cd078134152b/crates/line-tables-only>.
// Part of <https://github.com/rust-lang/rust/issues/122899> porting some backtrace tests to rustc.
// This test diverges from the original test in that it now uses a Rust library auxiliary because
// rustc now has `-Cdebuginfo=line-tables-only`.
//@ run-pass
//@ compile-flags: -Cstrip=none -Cdebuginfo=line-tables-only
//@ ignore-android FIXME #17520
//@ ignore-fuchsia Backtraces not symbolized
//@ ignore-emscripten Requires custom symbolization code
//@ ignore-ios needs the `.dSYM` files to be moved to the device
//@ ignore-tvos needs the `.dSYM` files to be moved to the device
//@ ignore-watchos needs the `.dSYM` files to be moved to the device
//@ ignore-visionos needs the `.dSYM` files to be moved to the device
//@ needs-unwind
//@ aux-build: line-tables-only-helper.rs
//@ edition: 2021
#![feature(backtrace_frames)]

extern crate line_tables_only_helper;

use std::backtrace::Backtrace;

fn assert_contains(
    backtrace: &Backtrace,
    expected_name: &str,
    expected_file: &str,
    expected_line: u32,
) {
    // The formatted frames look like this:
    // `{ fn: "line_tables_only_helper::backtrace_with_baz_in_it::<line_tables_only_helper::capture_backtrace::{closure#0}>", file: ".../tests/ui/backtrace/auxiliary/line-tables-only-helper.rs", line: 4 },
    // Make sure we match the right part when searching for the function name and line number.
    let expected_line_str = format!("line: {expected_line} ");
    eprintln!("{:#?}", backtrace);
    for frame in backtrace.frames() {
        // FIXME: we use string matching. Replace this by getting the actual data out of the frame,
        // once that is possible.
        let frame = format!("{:#?}", frame);
        if frame.contains(expected_name)
            && frame.contains(expected_file)
            && frame.contains(&expected_line_str)
        {
            return;
        }
    }
    panic!(
        "backtrace does not contain expected frame with name={expected_name}, file={expected_file}, line={expected_line}"
    );
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let backtrace = line_tables_only_helper::capture_backtrace();

    // FIXME(jieyouxu): for some forsaken reason on i686-msvc `foo` doesn't have an entry in the
    // line tables?
    // And with #143208 we also lost `bar` in the line tables.
    #[cfg(not(all(target_pointer_width = "32", target_env = "msvc")))]
    {
        assert_contains(&backtrace, "backtrace_with_foo_in_it", "line-tables-only-helper.rs", 12);
        assert_contains(&backtrace, "backtrace_with_bar_in_it", "line-tables-only-helper.rs", 8);
    }
    assert_contains(&backtrace, "backtrace_with_baz_in_it", "line-tables-only-helper.rs", 4);
}
