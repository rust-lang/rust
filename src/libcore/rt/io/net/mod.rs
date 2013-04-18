// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;

pub mod tcp;
pub mod udp;
pub mod ip;
#[cfg(unix)]
pub mod unix;
pub mod http;

/// A listener is a value that listens for connections
pub trait Listener<S> {
    /// Wait for and accept an incoming connection
    ///
    /// Returns `None` on timeout.
    ///
    /// # Failure
    ///
    /// Raises `io_error` condition. If the condition is handled,
    /// then `accept` returns `None`.
    fn accept(&mut self) -> Option<S>;
}
