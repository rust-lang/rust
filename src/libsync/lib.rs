// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Core concurrency-enabled mechanisms and primitives.
//!
//! This crate contains the implementations of Rust's core synchronization
//! primitives. This includes channels, mutexes, condition variables, etc.
//!
//! The interface of this crate is experimental, and it is not recommended to
//! use this crate specifically. Instead, its functionality is reexported
//! through `std::sync`.

#![crate_id = "sync#0.11.0-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/",
       html_playground_url = "http://play.rust-lang.org/")]
#![feature(phase, globs, macro_rules)]
#![deny(deprecated_owned_vector)]
#![deny(missing_doc)]
#![no_std]

#[cfg(stage0)]
#[phase(syntax, link)] extern crate core;
#[cfg(not(stage0))]
#[phase(plugin, link)] extern crate core;
extern crate alloc;
extern crate collections;
extern crate rustrt;

#[cfg(test)] extern crate test;
#[cfg(test)] extern crate native;
#[cfg(test, stage0)] #[phase(syntax, link)] extern crate std;
#[cfg(test, not(stage0))] #[phase(plugin, link)] extern crate std;

pub use alloc::arc::{Arc, Weak};
pub use lock::{Mutex, MutexGuard, Condvar, Barrier,
               RWLock, RWLockReadGuard, RWLockWriteGuard};

// The mutex/rwlock in this module are not meant for reexport
pub use raw::{Semaphore, SemaphoreGuard};

// Core building blocks for all primitives in this crate

pub mod atomics;

// Concurrent data structures

mod mpsc_intrusive;
pub mod spsc_queue;
pub mod mpsc_queue;
pub mod mpmc_bounded_queue;
pub mod deque;

// Low-level concurrency primitives

pub mod raw;
pub mod mutex;
pub mod one;

// Message-passing based communication

pub mod comm;

// Higher level primitives based on those above

mod lock;

#[cfg(not(test))]
mod std {
    pub use core::{fmt, option, cmp, clone};
}
