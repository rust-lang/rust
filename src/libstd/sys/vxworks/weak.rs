//! Support for "weak linkage" to symbols on Unix
//!
//! Some I/O operations we do in libstd require newer versions of OSes but we
//! need to maintain binary compatibility with older releases for now. In order
//! to use the new functionality when available we use this module for
//! detection.
//!
//! One option to use here is weak linkage, but that is unfortunately only
//! really workable on Linux. Hence, use dlsym to get the symbol value at
//! runtime. This is also done for compatibility with older versions of glibc,
//! and to avoid creating dependencies on GLIBC_PRIVATE symbols. It assumes that
//! we've been dynamically linked to the library the symbol comes from, but that
//! is currently always the case for things like libpthread/libc.
//!
//! A long time ago this used weak linkage for the __pthread_get_minstack
//! symbol, but that caused Debian to detect an unnecessarily strict versioned
//! dependency on libc6 (#23628).

use crate::ffi::CStr;
use crate::marker;
use crate::mem;
use crate::sync::atomic::{AtomicUsize, Ordering};

pub struct Weak<F> {
    name: &'static str,
    addr: AtomicUsize,
    _marker: marker::PhantomData<F>,
}

impl<F> Weak<F> {
    pub const fn new(name: &'static str) -> Weak<F> {
        Weak { name, addr: AtomicUsize::new(1), _marker: marker::PhantomData }
    }

    pub fn get(&self) -> Option<F> {
        assert_eq!(mem::size_of::<F>(), mem::size_of::<usize>());
        unsafe {
            if self.addr.load(Ordering::SeqCst) == 1 {
                self.addr.store(fetch(self.name), Ordering::SeqCst);
            }
            match self.addr.load(Ordering::SeqCst) {
                0 => None,
                addr => Some(mem::transmute_copy::<usize, F>(&addr)),
            }
        }
    }
}

unsafe fn fetch(name: &str) -> usize {
    let name = match CStr::from_bytes_with_nul(name.as_bytes()) {
        Ok(cstr) => cstr,
        Err(..) => return 0,
    };
    assert!(false, "FIXME: fetch");
    libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) as usize
}
