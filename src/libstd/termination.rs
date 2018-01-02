// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::Error;
#[cfg(target_arch = "wasm32")]
mod exit {
    pub const SUCCESS: i32 = 0;
    pub const FAILURE: i32 = 1;
}
#[cfg(not(target_arch = "wasm32"))]
mod exit {
    use libc;
    pub const SUCCESS: i32 = libc::EXIT_SUCCESS;
    pub const FAILURE: i32 = libc::EXIT_FAILURE;
}

/// A trait for implementing arbitrary return types in the `main` function.
///
/// The c-main function only supports to return integers as return type.
/// So, every type implementing the `Termination` trait has to be converted
/// to an integer.
///
/// The default implementations are returning `libc::EXIT_SUCCESS` to indicate
/// a successful execution. In case of a failure, `libc::EXIT_FAILURE` is returned.
#[cfg_attr(not(test), lang = "termination")]
#[unstable(feature = "termination_trait", issue = "43301")]
#[rustc_on_unimplemented =
  "`main` can only return types that implement {Termination}, not `{Self}`"]
pub trait Termination {
    /// Is called to get the representation of the value as status code.
    /// This status code is returned to the operating system.
    fn report(self) -> i32;
}

#[unstable(feature = "termination_trait", issue = "43301")]
impl Termination for () {
    fn report(self) -> i32 { exit::SUCCESS }
}

#[unstable(feature = "termination_trait", issue = "43301")]
impl<T: Termination, E: Error> Termination for Result<T, E> {
    fn report(self) -> i32 {
        match self {
            Ok(val) => val.report(),
            Err(err) => {
                print_error(err);
                exit::FAILURE
            }
        }
    }
}

#[unstable(feature = "termination_trait", issue = "43301")]
fn print_error<E: Error>(err: E) {
    eprintln!("Error: {}", err.description());

    if let Some(ref err) = err.cause() {
        eprintln!("Caused by: {}", err.description());
    }
}

#[unstable(feature = "termination_trait", issue = "43301")]
impl Termination for ! {
    fn report(self) -> i32 { unreachable!(); }
}

#[unstable(feature = "termination_trait", issue = "43301")]
impl Termination for bool {
    fn report(self) -> i32 {
        if self { exit::SUCCESS } else { exit::FAILURE }
    }
}

#[unstable(feature = "termination_trait", issue = "43301")]
impl Termination for i32 {
    fn report(self) -> i32 {
        self
    }
}
