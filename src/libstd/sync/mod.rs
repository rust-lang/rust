// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Useful synchronization primitives.
//!
//! ## The need for synchronization
//!
//! Conceptually, a Rust program is simply a series of operations which will
//! be executed on a computer. The timeline of events happening in the program
//! is consistent with the order of the operations in the code.
//!
//! Considering the following code, operating on some global static variables:
//!
//! ```rust
//! static mut A: u32 = 0;
//! static mut B: u32 = 0;
//! static mut C: u32 = 0;
//!
//! fn main() {
//!     unsafe {
//!         A = 3;
//!         B = 4;
//!         A = A + B;
//!         C = B;
//!         println!("{} {} {}", A, B, C);
//!         C = A;
//!     }
//! }
//! ```
//!
//! It appears _as if_ some variables stored in memory are changed, an addition
//! is performed, result is stored in `A` and the variable `C` is modified twice.
//! When only a single thread is involved, the results are as expected:
//! the line `7 4 4` gets printed.
//!
//! As for what happens behind the scenes, when optimizations are enabled the
//! final generated machine code might look very different from the code:
//!
//! - The first store to `C` might be moved before the store to `A` or `B`,
//!   _as if_ we had written `C = 4; A = 3; B = 4`.
//!
//! - Assignment of `A + B` to `A` might be removed, since the sum can be stored
//!   in a temporary location until it gets printed, with the global variable
//!   never getting updated.
//!
//! - The final result could be determined just by looking at the code at compile time,
//!   so [constant folding] might turn the whole block into a simple `println!("7 4 4")`.
//!
//! The compiler is allowed to perform any combination of these optimizations, as long
//! as the final optimized code, when executed, produces the same results as the one
//! without optimizations.
//!
//! When multiprocessing is involved (either multiple CPU cores, or multiple
//! physical CPUs), access to global variables (which are shared between threads)
//! could lead to nondeterministic results, **even if** compiler optimizations
//! are disabled.
//!
//! Note that thanks to Rust's safety guarantees, accessing global (static)
//! variables requires `unsafe` code, assuming we don't use any of the
//! synchronization primitives in this module.
//!
//! [constant folding]: https://en.wikipedia.org/wiki/Constant_folding
//!
//! ## Out-of-order execution
//!
//! Instructions can execute in a different order from the one we define, due to
//! various reasons:
//!
//! - **Compiler** reordering instructions: if the compiler can issue an
//!   instruction at an earlier point, it will try to do so. For example, it
//!   might hoist memory loads at the top of a code block, so that the CPU can
//!   start [prefetching] the values from memory.
//!
//!   In single-threaded scenarios, this can cause issues when writing
//!   signal handlers or certain kinds of low-level code.
//!   Use [compiler fences] to prevent this reordering.
//!
//! - **Single processor** executing instructions [out-of-order]: modern CPUs are
//!   capable of [superscalar] execution, i.e. multiple instructions might be
//!   executing at the same time, even though the machine code describes a
//!   sequential process.
//!
//!   This kind of reordering is handled transparently by the CPU.
//!
//! - **Multiprocessor** system, where multiple hardware threads run at the same time.
//!   In multi-threaded scenarios, you can use two kinds of primitives to deal
//!   with synchronization:
//!   - [memory fences] to ensure memory accesses are made visibile to other
//!     CPUs in the right order.
//!   - [atomic operations] to ensure simultaneous access to the same memory
//!     location doesn't lead to undefined behavior.
//!
//! [prefetching]: https://en.wikipedia.org/wiki/Cache_prefetching
//! [compiler fences]: crate::sync::atomic::compiler_fence
//! [out-of-order]: https://en.wikipedia.org/wiki/Out-of-order_execution
//! [superscalar]: https://en.wikipedia.org/wiki/Superscalar_processor
//! [memory fences]: crate::sync::atomic::fence
//! [atomic operations]: crate::sync::atomic
//!
//! ## Higher-level synchronization objects
//!
//! Most of the low-level synchronization primitives are quite error-prone and
//! inconvenient to use, which is why the standard library also exposes some
//! higher-level synchronization objects.
//!
//! These abstractions can be built out of lower-level primitives. For efficiency,
//! the sync objects in the standard library are usually implemented with help
//! from the operating system's kernel, which is able to reschedule the threads
//! while they are blocked on acquiring a lock.
//!
//! ## Efficiency
//!
//! Higher-level synchronization mechanisms are usually heavy-weight.
//! While most atomic operations can execute instantaneously, acquiring a
//! [`Mutex`] can involve blocking until another thread releases it.
//! For [`RwLock`], while any number of readers may acquire it without
//! blocking, each writer will have exclusive access.
//!
//! On the other hand, communication over [channels] can provide a fairly
//! high-level interface without sacrificing performance, at the cost of
//! somewhat more memory.
//!
//! The more synchronization exists between CPUs, the smaller the performance
//! gains from multithreading will be.
//!
//! [`Mutex`]: crate::sync::Mutex
//! [`RwLock`]: crate::sync::RwLock
//! [channels]: crate::sync::mpsc

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::sync::{Arc, Weak};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::sync::atomic;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::barrier::{Barrier, BarrierWaitResult};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::condvar::{Condvar, WaitTimeoutResult};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::mutex::{Mutex, MutexGuard};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::once::{Once, OnceState, ONCE_INIT};
#[stable(feature = "rust1", since = "1.0.0")]
pub use sys_common::poison::{PoisonError, TryLockError, TryLockResult, LockResult};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub mod mpsc;

mod barrier;
mod condvar;
mod mutex;
mod once;
mod rwlock;
