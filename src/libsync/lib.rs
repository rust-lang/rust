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

#[crate_id = "sync#0.10-pre"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[license = "MIT/ASL2"];
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];
#[feature(phase)];
#[allow(deprecated_owned_vector)]; // NOTE: remove after stage0

#[cfg(test)] #[phase(syntax, link)] extern crate log;

pub use arc::{Arc, MutexArc, RWArc, RWWriteMode, RWReadMode, ArcCondvar, CowArc};
pub use sync::{Mutex, RWLock, Condvar, Semaphore, RWLockWriteMode,
               RWLockReadMode, Barrier, one, mutex};
pub use comm::{DuplexStream, SyncSender, SyncReceiver, rendezvous, duplex};
pub use task_pool::TaskPool;
pub use future::Future;

mod arc;
mod sync;
mod comm;
mod task_pool;
mod future;
