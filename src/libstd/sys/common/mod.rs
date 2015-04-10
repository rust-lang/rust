// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]

use prelude::v1::*;

pub mod backtrace;
pub mod condvar;
pub mod mutex;
pub mod net2;
pub mod poison;
pub mod remutex;
pub mod rwlock;
pub mod stack;
pub mod thread;
pub mod thread_info;
pub mod thread_local;
pub mod wtf8;

// common error constructors

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types
#[doc(hidden)]
pub trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations
#[doc(hidden)]
pub trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}
