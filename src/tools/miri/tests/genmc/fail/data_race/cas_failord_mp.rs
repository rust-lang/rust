// Translated from GenMC's test `wrong/racy/cas-failord-mp`.
// The race on `DATA` only manifests if the CAS fails, and its failure ordering is properly modeled.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering::*;

use crate::genmc::*;

static X: AtomicI32 = AtomicI32::new(0);
static mut DATA: i32 = 0;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ids = [
            spawn_pthread_closure(|| {
                DATA = 42;
                X.store(1, Release);
            }),
            spawn_pthread_closure(|| {
                if X.compare_exchange(2, 3, Acquire, Relaxed) == Err(1) {
                    assert!(DATA == 42); //~ ERROR: Undefined Behavior: Non-atomic race
                }
            }),
        ];
        join_pthreads(ids);
    }
    0
}
