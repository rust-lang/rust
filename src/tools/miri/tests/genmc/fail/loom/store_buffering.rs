//@ revisions: non_genmc genmc
//@[genmc] compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2019 Carl Lerche

// This is the test `store_buffering` from `loom/test/litmus.rs`, adapted for Miri-GenMC.
// https://github.com/tokio-rs/loom/blob/dbf32b04bae821c64be44405a0bb72ca08741558/tests/litmus.rs

// This test shows the comparison between running Miri with or without GenMC.
// Without GenMC, Miri requires multiple iterations of the loop to detect the error.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    // For normal Miri, we need multiple repetitions, but GenMC should find the bug with only 1.
    const REPS: usize = if cfg!(non_genmc) { 128 } else { 1 };
    for _ in 0..REPS {
        // New atomics every iterations, so they don't influence each other.
        let x = AtomicUsize::new(0);
        let y = AtomicUsize::new(0);

        // FIXME(genmc,HACK): remove these initializing writes once Miri-GenMC supports mixed atomic-non-atomic accesses.
        x.store(0, Relaxed);
        y.store(0, Relaxed);

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
    }

    0
}
