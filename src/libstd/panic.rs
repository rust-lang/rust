// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Panic support in the standard library

#![unstable(feature = "std_panic", reason = "module recently added",
            issue = "27719")]

use thread::Result;

/// Invokes a closure, capturing the cause of panic if one occurs.
///
/// This function will return `Ok` with the closure's result if the closure
/// does not panic, and will return `Err(cause)` if the closure panics. The
/// `cause` returned is the object with which panic was originally invoked.
///
/// It is currently undefined behavior to unwind from Rust code into foreign
/// code, so this function is particularly useful when Rust is called from
/// another language (normally C). This can run arbitrary Rust code, capturing a
/// panic and allowing a graceful handling of the error.
///
/// It is **not** recommended to use this function for a general try/catch
/// mechanism. The `Result` type is more appropriate to use for functions that
/// can fail on a regular basis.
///
/// The closure provided is required to adhere to the `'static` bound to ensure
/// that it cannot reference data in the parent stack frame, mitigating problems
/// with exception safety.
///
/// # Examples
///
/// ```
/// #![feature(recover, std_panic)]
///
/// use std::panic;
///
/// let result = panic::recover(|| {
///     println!("hello!");
/// });
/// assert!(result.is_ok());
///
/// let result = panic::recover(|| {
///     panic!("oh no!");
/// });
/// assert!(result.is_err());
/// ```
#[unstable(feature = "recover", reason = "recent API addition",
           issue = "27719")]
pub fn recover<F, R>(f: F) -> Result<R> where F: FnOnce() -> R + 'static {
    let mut result = None;
    unsafe {
        let result = &mut result;
        try!(::rt::unwind::try(move || *result = Some(f())))
    }
    Ok(result.unwrap())
}
