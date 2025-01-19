//@ compile-flags:-Cstrip=none -Cdebuginfo=line-tables-only
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=1
//@ needs-unwind
//@ ignore-android FIXME #17520
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-emscripten no panic
//@ ignore-sgx Backtraces not symbolized
//@ ignore-fuchsia Backtraces not symbolized
//@ ignore-msvc the `__rust_{begin,end}_short_backtrace` symbols aren't reliable.

#![feature(rustc_attrs)]

// do not take effect since we already has a inner call of __rust_end_short_backtrace
#[rustc_end_short_backtrace]
fn first() {
    second();
}

#[rustc_end_short_backtrace]
fn second() {
    third(); // won't show up in backtrace
}

fn third() {
    fourth(); // won't show up in backtrace
}

fn fourth() {
    fifth(); // won't show up in backtrace
}

#[rustc_start_short_backtrace]
fn fifth() {
    sixth();
}

fn sixth() {
    seven();
}

fn seven() {
    panic!("debug!!!");
}

fn main() {
    first();
}
