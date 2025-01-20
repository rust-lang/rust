//@ compile-flags:--crate-type=lib
#![feature(rustc_attrs)]

#[inline] //~ ERROR invalid argument
#[rustc_start_short_backtrace]
fn foo() {}

#[inline] //~ ERROR invalid argument
#[rustc_end_short_backtrace]
fn bar() {}

#[inline(always)] //~ ERROR invalid argument
#[rustc_start_short_backtrace]
fn baz() {}

// ok
#[inline(never)]
#[rustc_start_short_backtrace]
fn meow() {}

// ok
#[rustc_start_short_backtrace]
fn mrrrp() {}

// ok
#[inline(always)]
#[rustc_skip_short_backtrace]
fn awawa() {}

// ok
#[inline]
#[rustc_skip_short_backtrace]
fn owo() {}
