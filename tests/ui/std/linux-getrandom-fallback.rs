//@ only-linux
//@ run-pass
//@ needs-unwind

#![feature(random)]
#![feature(rustc_private)]

extern crate libc;

use std::ffi::{c_void, c_uint, c_int};
use std::random::{SystemRng, Rng};
use std::cell::Cell;
use std::panic::catch_unwind;

thread_local! {
    static GETRANDOM_ERROR: Cell<c_int> = const { Cell::new(libc::ENOMEM) };
}

// This interposes the symbol defined in the libc to return an error when we
// want it to.
#[unsafe(no_mangle)]
fn getrandom(_buf: *const c_void, _size: usize, _flags: c_uint) -> isize {
    let errno = GETRANDOM_ERROR.get();
    unsafe { libc::__errno_location().write(errno) };
    -1
}

fn main() {
    // Step one:
    // Test that the interposed symbol actually gets used by having it return an
    // error that makes `SystemRng` panic.
    GETRANDOM_ERROR.set(libc::ENOMEM);
    catch_unwind(|| {
        let mut buf = [0; 16];
        SystemRng.fill_bytes(&mut buf);
    }).expect_err("SystemRng should panic upon receiving ENOMEM from the interposed getrandom");

    // Step two:
    // Emulate a missing `getrandom` by returning `ENOSYS`. This should excercise
    // the fallback code.
    GETRANDOM_ERROR.set(libc::ENOSYS);
    let mut buf = [0; 16];
    SystemRng.fill_bytes(&mut buf);

    // Smoke check that the buffer was actually filled. It is possible for this
    // to spuriously fail, but the likelyhood of that happening is 1 in 2^256.
    assert_ne!(buf, [0; 16]);

    // And lastly, check that the random pool has been initialized by manually
    // calling the getrandom syscall and checking that it does not block. This
    // is unlikely to catch any issues since the randomness pool is nearly always
    // initialized, but who knows.
    let r = unsafe {
        libc::syscall(libc::SYS_getrandom, buf.as_mut_ptr(), buf.len(), libc::GRND_NONBLOCK)
    };
    assert_ne!(r, -1);
}
