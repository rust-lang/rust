// no-prefer-dynamic

#![allow(dead_code)]

// check dtor calling order when casting enums.

use std::sync::atomic;
use std::sync::atomic::Ordering;
use std::mem;

enum E {
    A = 0,
    B = 1,
    C = 2
}

static FLAG: atomic::AtomicUsize = atomic::AtomicUsize::new(0);

impl Drop for E {
    fn drop(&mut self) {
        // avoid dtor loop
        unsafe { mem::forget(mem::replace(self, E::B)) };

        FLAG.store(FLAG.load(Ordering::SeqCst)+1, Ordering::SeqCst);
    }
}

fn main() {
    assert_eq!(FLAG.load(Ordering::SeqCst), 0);
    {
        let e = E::C;
        assert_eq!(e as u32, 2);
        assert_eq!(FLAG.load(Ordering::SeqCst), 0);
    }
    assert_eq!(FLAG.load(Ordering::SeqCst), 0);
}
