//@revisions: single multiple
//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@error-in-other-file: resource exhaustion

// Ensure that we emit a proper error if GenMC fails to fulfill an allocation.
// Two variants: one for a single large allocation, one for multiple ones
// that are individually below the limit, but together are too big.

#![no_main]

#[path = "../../../utils/genmc.rs"]
mod genmc;

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    if cfg!(multiple) {
        for _i in 1..8 {
            let _v = Vec::<u8>::with_capacity(1024 * 1024 * 1024);
        }
    } else {
        let _v = Vec::<u8>::with_capacity(8 * 1024 * 1024 * 1024);
    }
    0
}
