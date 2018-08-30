// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Test that #[derive(Copy, Clone)] produces a shallow copy
//! even when a member violates RFC 1521

use std::sync::atomic::{AtomicBool, ATOMIC_BOOL_INIT, Ordering};

/// A struct that pretends to be Copy, but actually does something
/// in its Clone impl
#[derive(Copy)]
struct Liar;

/// Static cooperating with the rogue Clone impl
static CLONED: AtomicBool = ATOMIC_BOOL_INIT;

impl Clone for Liar {
    fn clone(&self) -> Self {
        // this makes Clone vs Copy observable
        CLONED.store(true, Ordering::SeqCst);

        *self
    }
}

/// This struct is actually Copy... at least, it thinks it is!
#[derive(Copy, Clone)]
struct Innocent(Liar);

impl Innocent {
    fn new() -> Self {
        Innocent(Liar)
    }
}

fn main() {
    let _ = Innocent::new().clone();
    // if Innocent was byte-for-byte copied, CLONED will still be false
    assert!(!CLONED.load(Ordering::SeqCst));
}

