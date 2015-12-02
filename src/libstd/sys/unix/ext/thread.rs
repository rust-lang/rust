// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific extensions to primitives in the `std::process` module.

use os::unix::raw::{pthread_t};
use sys_common::{AsInner, IntoInner};
use thread::{JoinHandle};

#[unstable(feature = "thread_extensions", issue = "0")]
pub type RawId = pthread_t;

/// Unix-specific extensions to `std::thread::JoinHandle`
#[unstable(feature = "thread_extensions", issue = "0")]
pub trait JoinHandleExt {
    /// Extracts the raw pthread_t without taking ownership
    fn as_pthread_t(&self) -> RawId;
    /// Consumes the thread, returning the raw pthread_t
    ///
    /// This function **transfers ownership** of the underlying pthread_t to
    /// the caller. Callers are then the unique owners of the pthread_t and
    /// must either detech or join the pthread_t once it's no longer needed.
    fn into_pthread_t(self) -> RawId;
}

impl<T> JoinHandleExt for JoinHandle<T> {
    fn as_pthread_t(&self) -> RawId {
        self.as_inner().id() as RawId
    }
    fn into_pthread_t(self) -> RawId {
        self.into_inner().into_id() as RawId
    }
}
