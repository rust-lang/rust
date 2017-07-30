// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific extensions to primitives in the `std::thread` module.

#![stable(feature = "thread_extensions", since = "1.9.0")]

use sys_common::{AsInner, IntoInner};
use thread::JoinHandle;

#[stable(feature = "thread_extensions", since = "1.9.0")]
#[allow(deprecated)]
pub type RawPthread = usize;

/// Unix-specific extensions to `std::thread::JoinHandle`
#[stable(feature = "thread_extensions", since = "1.9.0")]
pub trait JoinHandleExt {
    /// Extracts the raw pthread_t without taking ownership
    #[stable(feature = "thread_extensions", since = "1.9.0")]
    fn as_pthread_t(&self) -> RawPthread;

    /// Consumes the thread, returning the raw pthread_t
    ///
    /// This function **transfers ownership** of the underlying pthread_t to
    /// the caller. Callers are then the unique owners of the pthread_t and
    /// must either detach or join the pthread_t once it's no longer needed.
    #[stable(feature = "thread_extensions", since = "1.9.0")]
    fn into_pthread_t(self) -> RawPthread;
}

#[stable(feature = "thread_extensions", since = "1.9.0")]
impl<T> JoinHandleExt for JoinHandle<T> {
    fn as_pthread_t(&self) -> RawPthread {
        self.as_inner().id() as RawPthread
    }

    fn into_pthread_t(self) -> RawPthread {
        self.into_inner().into_id() as RawPthread
    }
}
