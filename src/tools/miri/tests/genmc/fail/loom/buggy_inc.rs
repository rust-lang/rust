//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2019 Carl Lerche

// This is the test `checks_fail` from loom/test/smoke.rs adapted for Miri-GenMC.
// https://github.com/tokio-rs/loom/blob/dbf32b04bae821c64be44405a0bb72ca08741558/tests/smoke.rs

// This test checks that an incorrect implementation of an incrementing counter is detected.
// The counter behaves wrong if two threads try to increment at the same time (increments can be lost).

#![no_main]

#[cfg(not(any(non_genmc_std, genmc_std)))]
#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

struct BuggyInc {
    num: AtomicUsize,
}

impl BuggyInc {
    const fn new() -> BuggyInc {
        BuggyInc { num: AtomicUsize::new(0) }
    }

    fn inc(&self) {
        // The bug is here:
        // Another thread can increment `self.num` between the next two lines,
        // which is then overridden by this thread.
        let curr = self.num.load(Acquire);
        self.num.store(curr + 1, Release);
    }
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        static BUGGY_INC: BuggyInc = BuggyInc::new();
        let ids = [
            spawn_pthread_closure(|| {
                BUGGY_INC.inc();
            }),
            spawn_pthread_closure(|| {
                BUGGY_INC.inc();
            }),
        ];
        // Join so we can read the final values.
        join_pthreads(ids);

        // We check that we can detect the incorrect counter implementation:
        if 2 != BUGGY_INC.num.load(Relaxed) {
            std::process::abort(); //~ ERROR: abnormal termination
        }

        0
    }
}
