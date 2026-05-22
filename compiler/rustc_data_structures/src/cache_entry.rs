use std::cell::UnsafeCell;
use std::mem::{ManuallyDrop, MaybeUninit, needs_drop};
use std::sync::atomic::{self, AtomicU32, Ordering};

use rustc_thread_pool::current_num_threads;

use crate::sync::{DynSend, DynSync, Parker, Unparker};

pub enum GetOrStartError<'a, V, I> {
    InProgress(EntryInProgress<'a, V>),
    Interrupted(I),
}

pub enum GetError<I> {
    InProgress,
    Interrupted(I),
}

pub struct Status(AtomicU32);

impl Status {
    const EMPTY: u32 = 0;
    const POISONED: u32 = 1;

    // FIXME: consider using lower bits instead of high ones and swap bits' meaning with its
    // opposite to optimize immediate values on RISC architectures.

    // If set then lower status bits should represent which worker threads are waiting on this query
    const IN_PROGRESS_BIT: u32 = 1 << (u32::BITS - 1);
    // If set then lower status bits should represent associated DepNodeIndex
    const NOT_IN_PROGRESS_COMPLETE_BIT: u32 = 1 << (u32::BITS - 2);

    const NOT_IN_PROGRESS_COMPLETE_INDEX_MASK: u32 =
        !(Self::IN_PROGRESS_BIT | Self::NOT_IN_PROGRESS_COMPLETE_BIT);
    const IN_PROGRESS_THREAD_INDEX_MASK: u32 = !Self::IN_PROGRESS_BIT;

    const fn complete(index: u32) -> u32 {
        debug_assert!(index & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK == 0);
        index | Status::NOT_IN_PROGRESS_COMPLETE_BIT
    }

    pub fn waiter_threads(&self) -> u32 {
        let status = self.0.load(atomic::Ordering::Relaxed);
        assert!(status & Self::IN_PROGRESS_BIT != 0);
        status & Self::IN_PROGRESS_THREAD_INDEX_MASK
    }

    pub fn remove_waiter_threads(&self, thread_mask: u32) {
        assert!(
            thread_mask & Self::IN_PROGRESS_BIT == 0,
            "{} {}",
            thread_mask,
            current_num_threads()
        );
        let mut status = self.0.load(atomic::Ordering::Relaxed);
        loop {
            assert!(status & Self::IN_PROGRESS_BIT != 0);
            assert!(status & thread_mask == thread_mask);
            let res = self.0.compare_exchange_weak(
                status,
                status & !thread_mask,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
            );
            if let Err(new_status) = res {
                status = new_status;
            } else {
                break;
            }
        }
    }
}

const _: () = {
    if Status::IN_PROGRESS_THREAD_INDEX_MASK.count_ones() as usize
        != rustc_thread_pool::max_num_threads()
    {
        panic!();
    }
};

pub struct CacheEntry<V> {
    status: Status,
    value: UnsafeCell<MaybeUninit<V>>,
}

unsafe impl<V: Send + Sync> Sync for CacheEntry<V> {}
unsafe impl<V: DynSend + DynSync> DynSync for CacheEntry<V> {}

impl<V> Default for CacheEntry<V> {
    fn default() -> Self {
        CacheEntry::empty()
    }
}

impl<V> CacheEntry<V> {
    #[inline]
    pub const fn empty() -> Self {
        CacheEntry {
            status: Status(AtomicU32::new(0)),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    #[inline]
    pub const fn complete(index: u32, x: V) -> Self {
        CacheEntry {
            status: Status(AtomicU32::new(Status::complete(index))),
            value: UnsafeCell::new(MaybeUninit::new(x)),
        }
    }

    pub fn status(&self) -> &Status {
        &self.status
    }

    #[inline]
    pub fn get_or_start<P: Parker>(
        &self,
        parker: P,
    ) -> Result<(&V, u32), GetOrStartError<'_, V, P::Interrupt>> {
        // Should this load be relaxed and use an Acquire fence if complete (or poisoned)?
        let mut status = self.status.0.load(Ordering::Acquire);
        loop {
            if status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
                == Status::NOT_IN_PROGRESS_COMPLETE_BIT
            {
                return Ok(unsafe { self.assume_complete(status) });
            } else if status == Status::EMPTY {
                let res = self.status.0.compare_exchange_weak(
                    Status::EMPTY,
                    Status::IN_PROGRESS_BIT,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                match res {
                    Ok(_) => {
                        return Err(GetOrStartError::InProgress(EntryInProgress { entry: self }));
                    }
                    Err(new) => {
                        status = new;
                        continue;
                    }
                }
            } else {
                return self.wait(parker).map_err(GetOrStartError::Interrupted);
            }
        }
    }

    #[inline]
    pub fn try_start(&self) -> Option<EntryInProgress<'_, V>> {
        let mut status = self.status.0.load(Ordering::Acquire);
        loop {
            if status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
                == Status::NOT_IN_PROGRESS_COMPLETE_BIT
            {
                return None;
            } else if status == Status::EMPTY {
                let res = self.status.0.compare_exchange_weak(
                    Status::EMPTY,
                    Status::IN_PROGRESS_BIT,
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
            } else if status == Status::POISONED {
                panic!("panic propagation")
            } else {
                return None;
            }
        }
    }

    #[inline]
    pub fn get<P: Parker>(&self, parker: P) -> Result<(&V, u32), GetError<P::Interrupt>> {
        let status = self.status.0.load(Ordering::Acquire);
        if status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
            == Status::NOT_IN_PROGRESS_COMPLETE_BIT
        {
            Ok(unsafe { self.assume_complete(status) })
        } else if status == Status::EMPTY {
            Err(GetError::InProgress)
        } else {
            self.wait(parker).map_err(GetError::Interrupted)
        }
    }

    #[inline]
    pub fn get_finished(&self) -> Option<(&V, u32)> {
        let status = self.status.0.load(Ordering::Acquire);
        if status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
            == Status::NOT_IN_PROGRESS_COMPLETE_BIT
        {
            Some(unsafe { self.assume_complete(status) })
        } else if status == Status::EMPTY {
            None
        } else {
            Self::in_progress_or_poisoned_panic(status)
        }
    }

    #[cold]
    fn in_progress_or_poisoned_panic(status: u32) -> ! {
        if status == Status::POISONED {
            panic!("panic propagation")
        } else {
            panic!("Entry is unexpectedly in progress")
        }
    }

    #[cold]
    fn wait<P: Parker>(&self, parker: P) -> Result<(&V, u32), P::Interrupt> {
        // FIXME: try spinning, that's a good trick
        let mut status = self.status.0.load(Ordering::Relaxed);
        let mut did_park = false;
        parker.park(|thread_index| {
            let thread_bit = 1 << thread_index;
            debug_assert_ne!(status, Status::EMPTY);
            loop {
                if status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
                    == Status::NOT_IN_PROGRESS_COMPLETE_BIT
                {
                    break false;
                }
                if status == Status::POISONED {
                    break false;
                }
                // Empty status is ruled as this is the slow path
                debug_assert!(status & Status::IN_PROGRESS_BIT != 0);
                debug_assert!(status & thread_bit == 0);
                let res = self.status.0.compare_exchange_weak(
                    status,
                    status | thread_bit,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                if let Err(new) = res {
                    status = new;
                } else {
                    did_park = true;
                    break true;
                }
            }
        })?;
        let old_status = status;
        if did_park {
            status = self.status.0.load(Ordering::Acquire);
        } else {
            atomic::fence(Ordering::Acquire);
        }
        if status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
            == Status::NOT_IN_PROGRESS_COMPLETE_BIT
        {
            unsafe { Ok(self.assume_complete(status)) }
        } else {
            debug_assert_eq!(status, Status::POISONED, "old status = {old_status:x}");
            panic!("Propagating panic")
        }
    }

    #[inline]
    unsafe fn assume_complete(&self, status: u32) -> (&V, u32) {
        debug_assert_eq!(
            status & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK,
            Status::NOT_IN_PROGRESS_COMPLETE_BIT
        );
        (
            unsafe { (*self.value.get()).assume_init_ref() },
            status & Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK,
        )
    }

    unsafe fn complete_unchecked(&self, value: V, index: u32, unparker: impl Unparker) -> &V {
        debug_assert!(self.status.0.load(atomic::Ordering::Relaxed) & Status::IN_PROGRESS_BIT != 0);
        let value = unsafe { (*self.value.get()).write(value) };
        let status = self.status.0.swap(Status::complete(index), Ordering::Release);
        debug_assert!(status & Status::IN_PROGRESS_BIT != 0);
        let waiters = status & !Status::IN_PROGRESS_BIT;
        if waiters != 0 {
            for i in 0..(u32::BITS - 2) {
                if waiters & (1 << i) != 0 {
                    unparker.unpark(i as usize);
                }
            }
        }
        value
    }
}

impl<V> Drop for CacheEntry<V> {
    fn drop(&mut self) {
        if needs_drop::<V>() {
            debug_assert!(*self.status.0.get_mut() & Status::IN_PROGRESS_BIT == 0);
            if *self.status.0.get_mut() & !Status::NOT_IN_PROGRESS_COMPLETE_INDEX_MASK
                == Status::NOT_IN_PROGRESS_COMPLETE_BIT
            {
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

    pub fn complete(self, value: V, index: u32, unparker: impl Unparker) -> &'a V {
        let this = ManuallyDrop::new(self);
        unsafe { this.entry.complete_unchecked(value, index, unparker) }
    }
}

impl<'a, V> Drop for EntryInProgress<'a, V> {
    fn drop(&mut self) {
        self.entry.status.0.store(Status::POISONED, Ordering::Release);
    }
}
