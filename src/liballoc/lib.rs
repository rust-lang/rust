// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rust's core allocation library
//!
//! This is the lowest level library through which allocation in Rust can be
//! performed where the allocation is assumed to succeed. This library will
//! trigger a task failure when allocation fails.
//!
//! This library, like libcore, is not intended for general usage, but rather as
//! a building block of other libraries. The types and interfaces in this
//! library are reexported through the [standard library](../std/index.html),
//! and should not be used through this library.
//!
//! Currently, there are four major definitions in this library.
//!
//! ## Owned pointers
//!
//! The [`Box`](owned/index.html) type is the core owned pointer type in rust.
//! There can only be one owner of a `Box`, and the owner can decide to mutate
//! the contents.
//!
//! This type can be sent among tasks efficiently as the size of a `Box` value
//! is just a pointer. Tree-like data structures are often built on owned
//! pointers because each node often has only one owner, the parent.
//!
//! ## Reference counted pointers
//!
//! The [`Rc`](rc/index.html) type is a non-threadsafe reference-counted pointer
//! type intended for sharing memory within a task. An `Rc` pointer wraps a
//! type, `T`, and only allows access to `&T`, a shared reference.
//!
//! This type is useful when inherited mutability is too constraining for an
//! application (such as using `Box`), and is often paired with the `Cell` or
//! `RefCell` types in order to allow mutation.
//!
//! ## Atomically reference counted pointers
//!
//! The [`Arc`](arc/index.html) type is the threadsafe equivalent of the `Rc`
//! type. It provides all the same functionality of `Rc`, except it requires
//! that the contained type `T` is shareable. Additionally, `Arc<T>` is itself
//! sendable while `Rc<T>` is not.
//!
//! This types allows for shared access to the contained data, and is often
//! paired with synchronization primitives such as mutexes to allow mutation of
//! shared resources.
//!
//! ## Heap interfaces
//!
//! The [`heap`](heap/index.html) and [`libc_heap`](libc_heap/index.html)
//! modules are the unsafe interfaces to the underlying allocation systems. The
//! `heap` module is considered the default heap, and is not necessarily backed
//! by libc malloc/free.  The `libc_heap` module is defined to be wired up to
//! the system malloc/free.

#![crate_id = "alloc#0.11.0-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/")]

#![no_std]
#![feature(phase)]

#[cfg(stage0)]
#[phase(syntax, link)]
extern crate core;

#[cfg(not(stage0))]
#[phase(plugin, link)]
extern crate core;

extern crate libc;


// Allow testing this library

#[cfg(test)] extern crate debug;
#[cfg(test)] extern crate sync;
#[cfg(test)] extern crate native;
#[cfg(test, stage0)] #[phase(syntax, link)] extern crate std;
#[cfg(test, stage0)] #[phase(syntax, link)] extern crate log;
#[cfg(test, not(stage0))] #[phase(plugin, link)] extern crate std;
#[cfg(test, not(stage0))] #[phase(plugin, link)] extern crate log;

// Heaps provided for low-level allocation strategies

pub mod heap;
pub mod libc_heap;
pub mod util;

// Primitive types using the heaps above

#[cfg(not(test))]
pub mod owned;
pub mod arc;
pub mod rc;

// FIXME(#14344): When linking liballoc with libstd, this library will be linked
//                as an rlib (it only exists as an rlib). It turns out that an
//                optimized standard library doesn't actually use *any* symbols
//                from this library. Everything is inlined and optimized away.
//                This means that linkers will actually omit the object for this
//                file, even though it may be needed in the future.
//
//                To get around this for now, we define a dummy symbol which
//                will never get inlined so the stdlib can call it. The stdlib's
//                reference to this symbol will cause this library's object file
//                to get linked in to libstd successfully (the linker won't
//                optimize it out).
#[doc(hidden)]
pub fn fixme_14344_be_sure_to_link_to_collections() {}

#[cfg(not(test))]
#[doc(hidden)]
mod std {
    pub use core::fmt;
    pub use core::option;
}
