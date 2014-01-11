// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The macros for managing the Rust compiler's diagnostic codes.

#[macro_escape];

/// Register a diagnostic code at compile time
///
/// Registers a diagnostic code at compile time so it can be used later
/// for reporting a diagnostic. All diagnostic codes must be registered once
/// and only once before use (lexically) in `alert_*!` or `desc_diag!`
/// macros.
///
/// Registration of diagnostics ensures that only known diagnostic codes are
/// used in diagnostic messages and only known diagnostic codes have extended
/// descriptions. It also creates a central place where all existing
/// diagnostic codes (both in use and not) are stored so that it is easy
/// to pick the next available diagnostic code when creating a new one.
#[cfg(not(stage0))]
macro_rules! reg_diag (
    ($name: tt) => {
        __tt_map_insert!(DIAGNOSTIC_REGISTRY, $name, $name)
    }
)

#[cfg(stage0)]
macro_rules! reg_diag (
    ($name: tt) => {
    }
)

#[cfg(not(stage0))]
macro_rules! reg_diag_msg (
    ($name: tt, $msg: tt) => { {
        // Validate that the diagnostic code is registered
        let _ = stringify!(__tt_map_get_expr!(DIAGNOSTIC_REGISTRY, $name));
        // Insert the diagnostic message into the DIAGNOSTIC_MSG table
        // so that desc_diag can retrieve it later. This also prevents the
        // same diagnostic from being raised in two different places.
        mod insert { __tt_map_insert!(DIAGNOSTIC_MSG, $name, $msg) }
    } }
)

#[cfg(stage0)]
macro_rules! reg_diag_msg (
    ($name: tt, $msg: tt) => { {
    } }
)

macro_rules! report_diag (
    ($f: expr, $name: tt, $msg: expr, $($arg: expr), *) => { {
        reg_diag_msg!($name, $msg);
        let msg: &str = format!($msg, $($arg), *);
        $f(stringify!($name), msg);
    } };
    ($f: expr, $name: tt, $msg: expr) => { {
        reg_diag_msg!($name, $msg);
        $f(stringify!($name), $msg);
    } }
)

macro_rules! report_diag_sp (
    ($f: expr, $sp: expr, $name: tt, $msg: expr, $($arg: expr), *) => { {
        reg_diag_msg!($name, $msg);
        let msg: &str = format!($msg, $($arg), *);
        $f(sp, stringify!($name), msg);
    } };
    ($f: expr, $sp: expr, $name: tt, $msg: expr) => { {
        reg_diag_msg!($name, $msg);
        $f(sp, stringify!($name), $msg);
    } }
)

/// Raise a diagnostic at the 'fatal' level
///
/// Report a diagnostic, registering its message literal at compile time so that
/// it can be retreived later (at compile time) by the `desc_diag!` macro.
///
/// This must be called (lexically) before a `desc_diag` on the same diagnostic code.
macro_rules! alert_fatal (
    ($sess: expr, $name: tt, $msg: expr, $($arg: expr), *) => (
        report_diag!(|c, m| $sess.fatal_with_diagnostic_code(c, m),
                     $name, $msg, $($arg), *);
    );
    ($sess: expr, $name: tt, $msg: expr) => (
        report_diag!(|c, m| $sess.fatal_with_diagnostic_code(c, m),
                     $name, $msg);
    )
)

/// Raise a diagnostic at the 'error' level
macro_rules! alert_err (
    ($sess: expr, $name: tt, $msg: expr, $($arg: expr), *) => (
        report_diag!(|c, m| $sess.err_with_diagnostic_code(c, m),
                     $name, $msg, $($arg), *);
    );
    ($sess: expr, $name: tt, $msg: expr) => (
        report_diag!(|c, m| $sess.err_with_diagnostic_code(c, m),
                     $name, $msg);
    )
)

/// Raise a diagnostic at the 'error' level
macro_rules! alert_warn (
    ($sess: expr, $name: tt, $msg: expr, $($arg: expr), *) => (
        report_diag!(|c, m| $sess.warn_with_diagnostic_code(c, m),
                     $name, $msg, $($arg), *);
    );
    ($sess: expr, $name: tt, $msg: expr) => (
        report_diag!(|c, m| $sess.warn_with_diagnostic_code(c, m),
                     $name, $msg);
    )
)

/// Raise a diagnostic at the 'fatal' level
macro_rules! span_fatal (
    ($sess: expr, $sp: expr, $name: tt, $msg: expr, $($arg: expr), *) => (
        report_diag!(|c, m| $sess.span_fatal_with_diagnostic_code($sp, c, m),
                     $name, $msg, $($arg), *);
    );
    ($sess: expr, $sp: expr, $name: tt, $msg: expr) => (
        report_diag!(|c, m| $sess.span_fatal_with_diagnostic_code($sp, c, m),
                     $name, $msg);
    )
)

/// Raise a diagnostic at the 'error' level
macro_rules! span_err (
    ($sess: expr, $sp: expr, $name: tt, $msg: expr, $($arg: expr), *) => (
        report_diag!(|c, m| $sess.span_err_with_diagnostic_code($sp, c, m),
                     $name, $msg, $($arg), *);
    );
    ($sess: expr, $sp: expr, $name: tt, $msg: expr) => (
        report_diag!(|c, m| $sess.span_err_with_diagnostic_code($sp, c, m),
                     $name, $msg);
    )
)

/// Raise a diagnostic at the 'warning' level
macro_rules! span_warn (
    ($sess: expr, $sp: expr, $name: tt, $msg: expr, $($arg: expr), *) => (
        report_diag!(|c, m| $sess.span_warn_with_diagnostic_code($sp, c, m),
                     $name, $msg, $($arg), *);
    );
    ($sess: expr, $sp: expr, $name: tt, $msg: expr) => (
        report_diag!(|c, m| $sess.span_warn_with_diagnostic_code($sp, c, m),
                     $name, $msg);
    )
)

/// Describe a diagnostic code and return info about it at runtime
///
/// Returns a tuple of strings containing (the diagnostic code, the diagnostic
/// message reported for the code, the extended description of the diagnostic).
/// Repated calls to this macro can be used to provide extended documentation about
/// errors and to build up a database of diagnostic information.
#[cfg(not(stage0))]
macro_rules! desc_diag (
    ($name: tt, $desc: expr) => {
        (stringify!($name), __tt_map_get_expr!(DIAGNOSTIC_MSG, $name), $desc)
    }
)

#[cfg(stage0)]
macro_rules! desc_diag (
    ($name: tt, $desc: expr) => {
        (stringify!($name), "unknown", $desc)
    }
)
