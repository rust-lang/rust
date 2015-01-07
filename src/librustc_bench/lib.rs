// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support code for rustc's built in micro-benchmarking framework.
//!
//! See the [Testing Guide](../guide-testing.html) for more details.

#![crate_name = "rustc_bench"]
#![unstable]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/")]
#![feature(asm)]

use std::time::Duration;

/// A function that is opaque to the optimizer, to allow benchmarks to
/// pretend to use outputs to assist in avoiding dead-code
/// elimination.
///
/// This function is a no-op, and does not even read from `dummy`.
#[stable]
pub fn black_box<T>(dummy: T) -> T {
    // we need to "use" the argument in some way LLVM can't
    // introspect.
    unsafe {asm!("" : : "r"(&dummy))}
    dummy
}

/// Manager of the benchmarking runs.
///
/// This is feed into functions marked with `#[bench]` to allow for
/// set-up & tear-down before running a piece of code repeatedly via a
/// call to `iter`.
#[stable]
#[allow(missing_copy_implementations)]
pub struct Bencher {
    iterations: u64,
    dur: Duration,

    /// A field to indicate the number of bytes that each iteration of the
    /// benchmarking procedure consumed. When set to a nonzero value, the speed
    /// of consumption can be printed in MB/s
    #[stable]
    pub bytes: u64,
}

impl Bencher {
    #[experimental]
    pub fn new() -> Bencher {
        Bencher {
            iterations: 0,
            dur: Duration::seconds(0),
            bytes: 0,
        }
    }

    /// Callback for benchmark functions to run in their body.
    #[stable]
    pub fn iter<T, F>(&mut self, mut inner: F) where F: FnMut() -> T {
        self.dur = Duration::span(|| {
            let k = self.iterations;
            for _ in range(0u64, k) {
                black_box(inner());
            }
        });
    }

    #[experimental]
    pub fn set_iterations(&mut self, n: u64) { self.iterations = n; }

    #[experimental]
    pub fn iterations(&self) -> u64 { self.iterations }

    #[experimental]
    pub fn dur(&self) -> Duration { self.dur }
}
