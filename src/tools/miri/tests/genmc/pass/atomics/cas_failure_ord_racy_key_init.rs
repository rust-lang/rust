//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows -Zmiri-ignore-leaks

// Adapted from: `impl LazyKey`, `fn lazy_init`: rust/library/std/src/sys/thread_local/key/racy.rs
// Two threads race to initialize a key, which is just an index into an array in this test.
// The (Acquire, Release) orderings on the compare_exchange prevent any data races for reading from `VALUES[key]`.

// FIXME(genmc): GenMC does not model the failure ordering of compare_exchange currently.
// Miri thus upgrades the success ordering to prevent showing any false data races in cases like this one.
// Once GenMC supports the failure ordering, this test should work without the upgrading.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

const KEY_SENTVAL: usize = usize::MAX;

static KEY: AtomicUsize = AtomicUsize::new(KEY_SENTVAL);

static mut VALUES: [usize; 2] = [0, 0];

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let mut a = 0;
        let mut b = 0;
        let ids = [
            spawn_pthread_closure(|| {
                VALUES[0] = 42;
                let key = get_or_init(0);
                if key > 2 {
                    std::process::abort();
                }
                a = VALUES[key];
            }),
            spawn_pthread_closure(|| {
                VALUES[1] = 1234;
                let key = get_or_init(1);
                if key > 2 {
                    std::process::abort();
                }
                b = VALUES[key];
            }),
        ];
        join_pthreads(ids);
        if a != b {
            std::process::abort();
        }
    }
    0
}

fn get_or_init(key: usize) -> usize {
    match KEY.compare_exchange(KEY_SENTVAL, key, Release, Acquire) {
        // The CAS succeeded, so we've created the actual key.
        Ok(_) => key,
        // If someone beat us to the punch, use their key instead.
        // The `Acquire` failure ordering means we synchronized with the `Release` compare_exchange in the thread that wrote the other key.
        // We can thus read from `VALUES[other_key]` without a data race.
        Err(other_key) => other_key,
    }
}
