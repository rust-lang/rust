// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2019 Carl Lerche

// This is the test `store_buffering` from `loom/test/litmus.rs`, adapted for Miri-GenMC.
// https://github.com/tokio-rs/loom/blob/dbf32b04bae821c64be44405a0bb72ca08741558/tests/litmus.rs

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // For normal Miri, we need multiple repetitions, but GenMC should find the bug with only 1.

    let x = AtomicUsize::new(0);
    let y = AtomicUsize::new(0);

    let mut a: usize = 1234;
    let mut b: usize = 1234;
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                x.store(1, Relaxed);
                a = y.load(Relaxed)
            }),
            spawn_pthread_closure(|| {
                y.store(1, Relaxed);
                b = x.load(Relaxed)
            }),
        ];
        join_pthreads(ids);
    }
    if (a, b) == (0, 0) {
        std::process::abort(); //~ ERROR: abnormal termination
    }

    0
}
