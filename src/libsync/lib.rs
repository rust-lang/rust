// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Concurrency-enabled mechanisms and primitives.
 */

#![crate_id = "sync#0.11-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]
#![feature(phase)]
#![deny(deprecated_owned_vector)]

// #![deny(missing_doc)] // NOTE: uncomment after a stage0 snap
#![allow(missing_doc)] // NOTE: remove after a stage0 snap
#![allow(visible_private_types)] // NOTE: remove after a stage0 snap

#[cfg(test)]
#[phase(syntax, link)] extern crate log;

pub use comm::{DuplexStream, duplex};
pub use task_pool::TaskPool;
pub use future::Future;
pub use arc::{Arc, Weak};
pub use lock::{Mutex, MutexGuard, Condvar, Barrier,
               RWLock, RWLockReadGuard, RWLockWriteGuard};

// The mutex/rwlock in this module are not meant for reexport
pub use raw::{Semaphore, SemaphoreGuard};

mod arc;
mod comm;
mod future;
mod lock;
mod mpsc_intrusive;
mod task_pool;

pub mod raw;
pub mod mutex;
pub mod one;
