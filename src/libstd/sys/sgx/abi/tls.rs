use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::ptr;
use crate::mem;
use crate::cell::Cell;
use crate::num::NonZeroUsize;
use self::sync_bitset::*;

#[cfg(target_pointer_width="64")]
const USIZE_BITS: usize = 64;
const TLS_KEYS: usize = 128; // Same as POSIX minimum
const TLS_KEYS_BITSET_SIZE: usize = (TLS_KEYS + (USIZE_BITS - 1)) / USIZE_BITS;

#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx3abi3tls14TLS_KEY_IN_USEE"]
static TLS_KEY_IN_USE: SyncBitset = SYNC_BITSET_INIT;
macro_rules! dup {
    ((* $($exp:tt)*) $($val:tt)*) => (dup!( ($($exp)*) $($val)* $($val)* ));
    (() $($val:tt)*) => ([$($val),*])
}
#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx3abi3tls14TLS_DESTRUCTORE"]
static TLS_DESTRUCTOR: [AtomicUsize; TLS_KEYS] = dup!((* * * * * * *) (AtomicUsize::new(0)));

extern "C" {
    fn get_tls_ptr() -> *const u8;
    fn set_tls_ptr(tls: *const u8);
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Key(NonZeroUsize);

impl Key {
    fn to_index(self) -> usize {
        self.0.get() - 1
    }

    fn from_index(index: usize) -> Self {
        Key(NonZeroUsize::new(index + 1).unwrap())
    }

    pub fn as_usize(self) -> usize {
        self.0.get()
    }

    pub fn from_usize(index: usize) -> Self {
        Key(NonZeroUsize::new(index).unwrap())
    }
}

#[repr(C)]
pub struct Tls {
    data: [Cell<*mut u8>; TLS_KEYS]
}

pub struct ActiveTls<'a> {
    tls: &'a Tls
}

impl<'a> Drop for ActiveTls<'a> {
    fn drop(&mut self) {
        let value_with_destructor = |key: usize| {
            let ptr = TLS_DESTRUCTOR[key].load(Ordering::Relaxed);
            unsafe { mem::transmute::<_,Option<unsafe extern fn(*mut u8)>>(ptr) }
                .map(|dtor| (&self.tls.data[key], dtor))
        };

        let mut any_non_null_dtor = true;
        while any_non_null_dtor {
            any_non_null_dtor = false;
            for (value, dtor) in TLS_KEY_IN_USE.iter().filter_map(&value_with_destructor) {
                let value = value.replace(ptr::null_mut());
                if value != ptr::null_mut() {
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
        set_tls_ptr(self as *const Tls as _);
        ActiveTls { tls: self }
    }

    #[allow(unused)]
    pub unsafe fn activate_persistent(self: Box<Self>) {
        set_tls_ptr((&*self) as *const Tls as _);
        mem::forget(self);
    }

    unsafe fn current<'a>() -> &'a Tls {
        &*(get_tls_ptr() as *const Tls)
    }

    pub fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
        let index = if let Some(index) = TLS_KEY_IN_USE.set() {
            index
        } else {
            rtabort!("TLS limit exceeded")
        };
        TLS_DESTRUCTOR[index].store(dtor.map_or(0, |f| f as usize), Ordering::Relaxed);
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

mod sync_bitset {
    use crate::sync::atomic::{AtomicUsize, Ordering};
    use crate::iter::{Enumerate, Peekable};
    use crate::slice::Iter;
    use super::{TLS_KEYS_BITSET_SIZE, USIZE_BITS};

    /// A bitset that can be used synchronously.
    pub(super) struct SyncBitset([AtomicUsize; TLS_KEYS_BITSET_SIZE]);

    pub(super) const SYNC_BITSET_INIT: SyncBitset =
        SyncBitset([AtomicUsize::new(0), AtomicUsize::new(0)]);

    impl SyncBitset {
        pub fn get(&self, index: usize) -> bool {
            let (hi, lo) = Self::split(index);
            (self.0[hi].load(Ordering::Relaxed) & lo) != 0
        }

        /// Not atomic.
        pub fn iter(&self) -> SyncBitsetIter<'_> {
            SyncBitsetIter {
                iter: self.0.iter().enumerate().peekable(),
                elem_idx: 0,
            }
        }

        pub fn clear(&self, index: usize) {
            let (hi, lo) = Self::split(index);
            self.0[hi].fetch_and(!lo, Ordering::Relaxed);
        }

        /// Sets any unset bit. Not atomic. Returns `None` if all bits were
        /// observed to be set.
        pub fn set(&self) -> Option<usize> {
            'elems: for (idx, elem) in self.0.iter().enumerate() {
                let mut current = elem.load(Ordering::Relaxed);
                loop {
                    if 0 == !current {
                        continue 'elems;
                    }
                    let trailing_ones = (!current).trailing_zeros() as usize;
                    match elem.compare_exchange(
                        current,
                        current | (1 << trailing_ones),
                        Ordering::AcqRel,
                        Ordering::Relaxed
                    ) {
                        Ok(_) => return Some(idx * USIZE_BITS + trailing_ones),
                        Err(previous) => current = previous,
                    }
                }
            }
            None
        }

        fn split(index: usize) -> (usize, usize) {
            (index / USIZE_BITS, 1 << (index % USIZE_BITS))
        }
    }

    pub(super) struct SyncBitsetIter<'a> {
        iter: Peekable<Enumerate<Iter<'a, AtomicUsize>>>,
        elem_idx: usize,
    }

    impl<'a> Iterator for SyncBitsetIter<'a> {
        type Item = usize;

        fn next(&mut self) -> Option<usize> {
            self.iter.peek().cloned().and_then(|(idx, elem)| {
                let elem = elem.load(Ordering::Relaxed);
                let low_mask = (1 << self.elem_idx) - 1;
                let next = elem & !low_mask;
                let next_idx = next.trailing_zeros() as usize;
                self.elem_idx = next_idx + 1;
                if self.elem_idx >= 64 {
                    self.elem_idx = 0;
                    self.iter.next();
                }
                match next_idx {
                    64 => self.next(),
                    _ => Some(idx * USIZE_BITS + next_idx),
                }
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn test_data(bitset: [usize; 2], bit_indices: &[usize]) {
            let set = SyncBitset([AtomicUsize::new(bitset[0]), AtomicUsize::new(bitset[1])]);
            assert_eq!(set.iter().collect::<Vec<_>>(), bit_indices);
            for &i in bit_indices {
                assert!(set.get(i));
            }
        }

        #[test]
        fn iter() {
            test_data([0b0110_1001, 0], &[0, 3, 5, 6]);
            test_data([0x8000_0000_0000_0000, 0x8000_0000_0000_0001], &[63, 64, 127]);
            test_data([0, 0], &[]);
        }

        #[test]
        fn set_get_clear() {
            let set = SYNC_BITSET_INIT;
            let key = set.set().unwrap();
            assert!(set.get(key));
            set.clear(key);
            assert!(!set.get(key));
        }
    }
}
