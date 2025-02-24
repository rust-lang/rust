//@ compile-flags:-Cstrip=none
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=1
//@ needs-unwind
//@ ignore-android FIXME #17520
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-sgx Backtraces not symbolized
//@ ignore-fuchsia Backtraces not symbolized
//@ ignore-msvc the `__rust_{begin,end}_short_backtrace` symbols aren't reliable.
// This is needed to avoid test output differences across std being built with v0 symbols vs legacy
// symbols.
//@ normalize-stderr: "begin_panic::<&str>" -> "begin_panic"
// This variant occurs on macOS with `rust.debuginfo-level = "line-tables-only"` (#133997)
//@ normalize-stderr: " begin_panic<&str>" -> " std::panicking::begin_panic"
// And this is for differences between std with and without debuginfo.
//@ normalize-stderr: "\n +at [^\n]+" -> ""

/// This test case make sure that we can have multiple pairs of `__rust_{begin,end}_short_backtrace`

#[inline(never)]
fn __rust_begin_short_backtrace<T, F: FnOnce() -> T>(f: F) -> T {
    let result = f();
    std::hint::black_box(result)
}

#[inline(never)]
fn __rust_end_short_backtrace<T, F: FnOnce() -> T>(f: F) -> T {
    let result = f();
    std::hint::black_box(result)
}

fn first() {
    __rust_end_short_backtrace(|| second());
}

fn second() {
    third(); // won't show up
}

fn third() {
    fourth(); // won't show up
}

fn fourth() {
    __rust_begin_short_backtrace(|| fifth());
}

fn fifth() {
    __rust_end_short_backtrace(|| sixth());
}

fn sixth() {
    seven(); // won't show up
}

fn seven() {
    __rust_begin_short_backtrace(|| eight());
}

fn eight() {
    panic!("debug!!!");
}

fn main() {
    first();
}
