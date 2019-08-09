//! Panic support for libcore
//!
//! The core library cannot define panicking, but it does *declare* panicking. This
//! means that the functions inside of libcore are allowed to panic, but to be
//! useful an upstream crate must define panicking for libcore to use. The current
//! interface for panicking is:
//!
//! ```
//! # use std::fmt;
//! fn panic_impl(fmt: fmt::Arguments, file_line_col: &(&'static str, u32, u32)) -> !
//! # { loop {} }
//! ```
//!
//! This definition allows for panicking with any general message, but it does not
//! allow for failing with a `Box<Any>` value. The reason for this is that libcore
//! is not allowed to allocate.
//!
//! This module contains a few other panicking functions, but these are just the
//! necessary lang items for the compiler. All panics are funneled through this
//! one function. Currently, the actual symbol is declared in the standard
//! library, but the location of this may change over time.

#![allow(dead_code, missing_docs)]
#![unstable(feature = "core_panic",
            reason = "internal details of the implementation of the `panic!` \
                      and related macros",
            issue = "0")]

use crate::fmt;
use crate::panic::{Location, PanicInfo};

#[cold]
// never inline unless panic_immediate_abort to avoid code
// bloat at the call sites as much as possible
#[cfg_attr(not(feature="panic_immediate_abort"),inline(never))]
#[lang = "panic"]
pub fn panic(expr_file_line_col: &(&'static str, &'static str, u32, u32)) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        unsafe { super::intrinsics::abort() }
    }

    // Use Arguments::new_v1 instead of format_args!("{}", expr) to potentially
    // reduce size overhead. The format_args! macro uses str's Display trait to
    // write expr, which calls Formatter::pad, which must accommodate string
    // truncation and padding (even though none is used here). Using
    // Arguments::new_v1 may allow the compiler to omit Formatter::pad from the
    // output binary, saving up to a few kilobytes.
    let (expr, file, line, col) = *expr_file_line_col;
    panic_fmt(fmt::Arguments::new_v1(&[expr], &[]), &(file, line, col))
}

#[cold]
#[cfg_attr(not(feature="panic_immediate_abort"),inline(never))]
#[lang = "panic_bounds_check"]
fn panic_bounds_check(file_line_col: &(&'static str, u32, u32),
                     index: usize, len: usize) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        unsafe { super::intrinsics::abort() }
    }

    panic_fmt(format_args!("index out of bounds: the len is {} but the index is {}",
                           len, index), file_line_col)
}

#[cold]
#[cfg_attr(not(feature="panic_immediate_abort"),inline(never))]
#[cfg_attr(    feature="panic_immediate_abort" ,inline)]
pub fn panic_fmt(fmt: fmt::Arguments<'_>, file_line_col: &(&'static str, u32, u32)) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        unsafe { super::intrinsics::abort() }
    }

    // NOTE This function never crosses the FFI boundary; it's a Rust-to-Rust call
    #[allow(improper_ctypes)] // PanicInfo contains a trait object which is not FFI safe
    extern "Rust" {
        #[lang = "panic_impl"]
        fn panic_impl(pi: &PanicInfo<'_>) -> !;
    }

    let (file, line, col) = *file_line_col;
    let pi = PanicInfo::internal_constructor(
        Some(&fmt),
        Location::internal_constructor(file, line, col),
    );
    unsafe { panic_impl(&pi) }
}
