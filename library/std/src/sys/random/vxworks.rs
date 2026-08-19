use crate::io::BorrowedCursor;
use crate::sync::atomic::Ordering::Relaxed;
use crate::sync::atomic::{Atomic, AtomicBool};

static RNG_INIT: Atomic<bool> = AtomicBool::new(false);

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    while !RNG_INIT.load(Relaxed) {
        let ret = unsafe { libc::randSecure() };
        if ret < 0 {
            panic!("failed to generate random data");
        } else if ret > 0 {
            RNG_INIT.store(true, Relaxed);
            break;
        }

        unsafe { libc::usleep(10) };
    }

    while cursor.capacity() != 0 {
        let len = cursor.capacity().try_into().unwrap_or(libc::c_int::MAX);
        let ret = unsafe { libc::randABytes(cursor.as_mut().as_mut_ptr().cast(), len) };
        assert!(ret >= 0, "failed to generate random data");
        // SAFETY: We've just initialized `len` bytes
        unsafe {
            cursor.advance(len as usize);
        }
    }
}
