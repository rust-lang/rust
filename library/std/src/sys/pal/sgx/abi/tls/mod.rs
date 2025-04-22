mod sync_bitset;

use self::sync_bitset::*;
use crate::cell::Cell;
use crate::num::NonZero;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::{mem, ptr};

#[cfg(target_pointer_width = "64")]
const USIZE_BITS: usize = 64;
const TLS_KEYS: usize = 128; // Same as POSIX minimum
const TLS_KEYS_BITSET_SIZE: usize = (TLS_KEYS + (USIZE_BITS - 1)) / USIZE_BITS;

// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3pal3sgx3abi3tls14TLS_KEY_IN_USEE")]
static TLS_KEY_IN_USE: SyncBitset = SYNC_BITSET_INIT;
macro_rules! dup {
    ((* $($exp:tt)*) $($val:tt)*) => (dup!( ($($exp)*) $($val)* $($val)* ));
    (() $($val:tt)*) => ([$($val),*])
}
// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3pal3sgx3abi3tls14TLS_DESTRUCTORE")]
static TLS_DESTRUCTOR: [AtomicUsize; TLS_KEYS] = dup!((* * * * * * *) (AtomicUsize::new(0)));

unsafe extern "C" {
    fn get_tls_ptr() -> *const u8;
    fn set_tls_ptr(tls: *const u8);
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Key(NonZero<usize>);

impl Key {
    fn to_index(self) -> usize {
        self.0.get() - 1
    }

    fn from_index(index: usize) -> Self {
        Key(NonZero::new(index + 1).unwrap())
    }

    pub fn as_usize(self) -> usize {
        self.0.get()
    }

    pub fn from_usize(index: usize) -> Self {
        Key(NonZero::new(index).unwrap())
    }
}

#[repr(C)]
pub struct Tls {
    data: [Cell<*mut u8>; TLS_KEYS],
}

pub struct ActiveTls<'a> {
    tls: &'a Tls,
}

impl<'a> Drop for ActiveTls<'a> {
    fn drop(&mut self) {
        let value_with_destructor = |key: usize| {
            let ptr = TLS_DESTRUCTOR[key].load(Ordering::Relaxed);
            unsafe { mem::transmute::<_, Option<unsafe extern "C" fn(*mut u8)>>(ptr) }
                .map(|dtor| (&self.tls.data[key], dtor))
        };

        let mut any_non_null_dtor = true;
        while any_non_null_dtor {
            any_non_null_dtor = false;
            for (value, dtor) in TLS_KEY_IN_USE.iter().filter_map(&value_with_destructor) {
                let value = value.replace(ptr::null_mut());
                if !value.is_null() {
                    any_non_null_dtor = true;
                    unsafe { dtor(value) }
                }
            }
        }
    }
}

impl Tls {
    pub fn new() -> Tls {
        Tls { data: dup!((* * * * * * *) (Cell::new(ptr::null_mut()))) }
    }

    pub unsafe fn activate(&self) -> ActiveTls<'_> {
        // FIXME: Needs safety information. See entry.S for `set_tls_ptr` definition.
        unsafe { set_tls_ptr(self as *const Tls as _) };
        ActiveTls { tls: self }
    }

    #[allow(unused)]
    pub unsafe fn activate_persistent(self: Box<Self>) {
        // FIXME: Needs safety information. See entry.S for `set_tls_ptr` definition.
        let ptr = Box::into_raw(self).cast_const().cast::<u8>();
        unsafe { set_tls_ptr(ptr) };
    }

    unsafe fn current<'a>() -> &'a Tls {
        // FIXME: Needs safety information. See entry.S for `set_tls_ptr` definition.
        unsafe { &*(get_tls_ptr() as *const Tls) }
    }

    pub fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
        let index = if let Some(index) = TLS_KEY_IN_USE.set() {
            index
        } else {
            rtabort!("TLS limit exceeded")
        };
        TLS_DESTRUCTOR[index].store(dtor.map_or(0, |f| f as usize), Ordering::Relaxed);
        unsafe { Self::current() }.data[index].set(ptr::null_mut());
        Key::from_index(index)
    }

    pub fn set(key: Key, value: *mut u8) {
        let index = key.to_index();
        rtassert!(TLS_KEY_IN_USE.get(index));
        unsafe { Self::current() }.data[index].set(value);
    }

    pub fn get(key: Key) -> *mut u8 {
        let index = key.to_index();
        rtassert!(TLS_KEY_IN_USE.get(index));
        unsafe { Self::current() }.data[index].get()
    }

    pub fn destroy(key: Key) {
        TLS_KEY_IN_USE.clear(key.to_index());
    }
}
