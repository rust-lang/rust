use std::cell::UnsafeCell;
use std::mem::{ManuallyDrop, MaybeUninit, needs_drop};
use std::sync::atomic::{self, AtomicU32, Ordering};

use crate::sync::{DynSend, DynSync};

pub struct CacheEntry<V> {
    status: AtomicU32,
    value: UnsafeCell<MaybeUninit<V>>,
}

unsafe impl<V: Send> Sync for CacheEntry<V> {}
unsafe impl<V: DynSend> DynSync for CacheEntry<V> {}

impl<V> Default for CacheEntry<V> {
    fn default() -> Self {
        CacheEntry::empty()
    }
}

impl<V> CacheEntry<V> {
    const EMPTY_STATUS: u32 = 0;
    const POISONED_STATUS: u32 = 1;

    // FIXME: consider using lower bits instead of high ones and swap bits' meaning with its
    // opposite to optimize immediate values on RISC architectures.

    // If set then lower status bits should represent associated DepNodeIndex
    const COMPLETE_BIT: u32 = 1 << (u32::BITS - 1);
    // If set then lower status bits should represent which worker threads are waiting on this query
    const IN_PROGRESS_BIT: u32 = 1 << (u32::BITS - 2);
    const CONTROL_MASK: u32 = Self::COMPLETE_BIT | Self::IN_PROGRESS_BIT;

    const fn complete_status(index: u32) -> u32 {
        debug_assert!(index & Self::CONTROL_MASK == 0);
        index | Self::COMPLETE_BIT
    }

    pub const fn empty() -> Self {
        CacheEntry { status: AtomicU32::new(0), value: UnsafeCell::new(MaybeUninit::uninit()) }
    }

    #[inline]
    pub fn get_or_start(&self) -> Result<(&V, u32), EntryInProgress<'_, V>> {
        let mut status = self.status.load(Ordering::Acquire);
        loop {
            if status & Self::COMPLETE_BIT != 0 {
                return Ok(unsafe { self.assume_complete(status) });
            } else if status == Self::EMPTY_STATUS {
                let res = self.status.compare_exchange_weak(
                    Self::EMPTY_STATUS,
                    Self::IN_PROGRESS_BIT,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                match res {
                    Ok(_) => return Err(EntryInProgress { entry: self }),
                    Err(new) => {
                        status = new;
                        continue;
                    }
                }
            } else {
                return Ok(self.wait());
            }
        }
    }

    #[inline]
    pub fn try_start(&self) -> Option<EntryInProgress<'_, V>> {
        let mut status = self.status.load(Ordering::Acquire);
        loop {
            if status & Self::COMPLETE_BIT != 0 {
                return None;
            } else if status == Self::EMPTY_STATUS {
                let res = self.status.compare_exchange_weak(
                    Self::EMPTY_STATUS,
                    Self::IN_PROGRESS_BIT,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                match res {
                    Ok(_) => return Some(EntryInProgress { entry: self }),
                    Err(new) => {
                        status = new;
                        continue;
                    }
                }
            } else if status == Self::POISONED_STATUS {
                panic!("panic propagation")
            } else {
                return None;
            }
        }
    }

    #[inline]
    pub fn get(&self) -> Option<(&V, u32)> {
        let status = self.status.load(Ordering::Acquire);
        if status & Self::COMPLETE_BIT != 0 {
            Some(unsafe { self.assume_complete(status) })
        } else if status == Self::EMPTY_STATUS {
            None
        } else {
            Some(self.wait())
        }
    }

    #[inline]
    pub fn get_finished(&self) -> Option<(&V, u32)> {
        let status = self.status.load(Ordering::Acquire);
        if status & Self::COMPLETE_BIT != 0 {
            Some(unsafe { self.assume_complete(status) })
        } else if status == Self::EMPTY_STATUS {
            None
        } else {
            Self::in_progress_or_poisoned_panic(status)
        }
    }

    #[cold]
    fn in_progress_or_poisoned_panic(status: u32) -> ! {
        if status == Self::POISONED_STATUS {
            panic!("panic propagation")
        } else {
            panic!("Entry is unexpectedly in progress")
        }
    }

    #[cold]
    fn wait(&self) -> (&V, u32) {
        // FIXME: try spinning, that's a good trick
        let mut status = self.status.load(Ordering::Relaxed);
        let did_park = rustc_thread_pool::park(|thread_index| {
            let thread_bit = 1 << thread_index;
            loop {
                if status & Self::COMPLETE_BIT != 0 {
                    break false;
                }
                if status == Self::POISONED_STATUS {
                    break false;
                }
                // Empty status is ruled as this is the slow path
                debug_assert!(status & Self::IN_PROGRESS_BIT != 0);
                debug_assert!(status & thread_bit == 0);
                let res = self.status.compare_exchange_weak(
                    status,
                    status | thread_bit,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                if let Err(new) = res {
                    status = new;
                } else {
                    break true;
                }
            }
        });
        let old_status = status;
        if did_park {
            status = self.status.load(Ordering::Acquire);
        } else {
            atomic::fence(Ordering::Acquire);
        }
        if status & Self::COMPLETE_BIT != 0 {
            unsafe { self.assume_complete(status) }
        } else {
            debug_assert_eq!(status, Self::POISONED_STATUS, "old status = {old_status:x}");
            panic!("Propagating panic")
        }
    }

    #[inline]
    unsafe fn assume_complete(&self, status: u32) -> (&V, u32) {
        (unsafe { (*self.value.get()).assume_init_ref() }, status & !Self::COMPLETE_BIT)
    }

    unsafe fn complete_unchecked(&self, value: V, index: u32) -> &V {
        debug_assert!(self.status.load(atomic::Ordering::Relaxed) & Self::IN_PROGRESS_BIT != 0);
        let value = unsafe { (*self.value.get()).write(value) };
        let status = self.status.swap(Self::complete_status(index), Ordering::Release);
        debug_assert!(status & Self::CONTROL_MASK == Self::IN_PROGRESS_BIT);
        let waiters = status & !Self::IN_PROGRESS_BIT;
        if waiters != 0 {
            for i in 0..(u32::BITS - 2) {
                if waiters & (1 << i) != 0 {
                    rustc_thread_pool::unpark(i as usize);
                }
            }
        }
        value
    }
}

impl<V> Drop for CacheEntry<V> {
    fn drop(&mut self) {
        if needs_drop::<V>() {
            debug_assert!(*self.status.get_mut() & Self::IN_PROGRESS_BIT == 0);
            if *self.status.get_mut() & Self::COMPLETE_BIT != 0 {
                unsafe { self.value.get_mut().assume_init_drop() };
            }
        }
    }
}

pub struct EntryInProgress<'a, V> {
    entry: &'a CacheEntry<V>,
}

impl<'a, V> EntryInProgress<'a, V> {
    pub fn entry(&self) -> &'a CacheEntry<V> {
        self.entry
    }

    pub fn complete(self, value: V, index: u32) -> &'a V {
        let this = ManuallyDrop::new(self);
        unsafe { this.entry.complete_unchecked(value, index) }
    }
}

impl<'a, V> Drop for EntryInProgress<'a, V> {
    fn drop(&mut self) {
        self.entry.status.store(CacheEntry::<V>::POISONED_STATUS, Ordering::Release);
    }
}
