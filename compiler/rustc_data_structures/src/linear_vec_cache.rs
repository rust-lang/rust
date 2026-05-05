use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use rustc_index::Idx;

use crate::marker::{DynSend, DynSync};
use crate::sync::Lock;

mod vmem;

#[cfg(test)]
mod tests;

const LINEAR_MAX_SLOTS: u64 = (u32::MAX as u64) + 1;
const LINEAR_COMMIT_AHEAD_PAGES: usize = 128;

#[inline(always)]
const fn div_ceil(lhs: usize, rhs: usize) -> usize {
    lhs / rhs + ((lhs % rhs != 0) as usize)
}

#[inline(always)]
fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    let remainder = value % multiple;
    if remainder == 0 { value } else { value.checked_add(multiple - remainder).unwrap() }
}

struct Slot<T> {
    // We never construct &Slot<T> so it's fine for this to not be in an UnsafeCell.
    value: T,
    // This is both an index and a once-lock.
    //
    // 0: not yet initialized.
    // 1: lock held, initializing.
    // 2..u32::MAX - 2: initialized.
    index_and_lock: AtomicU32,
}

struct LinearStorage<T> {
    ptr: NonNull<Slot<T>>,
    committed_slots: AtomicUsize,
    committed_pages: Lock<usize>,
}

impl<T> LinearStorage<T> {
    fn max_index() -> usize {
        usize::try_from(LINEAR_MAX_SLOTS.saturating_sub(1)).unwrap_or(usize::MAX)
    }

    fn mapping_len_bytes() -> usize {
        let value_size = std::mem::size_of::<Slot<T>>();
        if value_size == 0 || LINEAR_MAX_SLOTS == 0 {
            0
        } else {
            let total_bytes = (value_size as u128).checked_mul(LINEAR_MAX_SLOTS as u128).unwrap();
            let total_bytes = usize::try_from(total_bytes).ok().unwrap_or_else(|| {
                panic!("LinearStorage cannot reserve requested virtual memory on this target")
            });
            round_up_to_multiple(total_bytes, vmem::page_size())
        }
    }

    fn new() -> Self {
        let len_bytes = Self::mapping_len_bytes();

        let ptr = if len_bytes == 0 {
            NonNull::dangling()
        } else {
            NonNull::new(
                vmem::reserve(len_bytes)
                    .unwrap_or_else(|err| panic!("failed to reserve linear storage mapping: {err}"))
                    .cast::<Slot<T>>(),
            )
            .unwrap()
        };

        LinearStorage { ptr, committed_slots: AtomicUsize::new(0), committed_pages: Lock::new(0) }
    }

    #[inline(always)]
    fn slot_ptr(&self, index: usize) -> *mut Slot<T> {
        debug_assert!(index <= Self::max_index());
        unsafe { self.ptr.as_ptr().add(index) }
    }

    #[inline(always)]
    fn pages_needed_for_index(index: usize) -> usize {
        let value_size = std::mem::size_of::<Slot<T>>();
        if value_size == 0 {
            0
        } else {
            let required_bytes = index.checked_add(1).unwrap().checked_mul(value_size).unwrap();
            div_ceil(required_bytes, vmem::page_size())
        }
    }

    #[inline(always)]
    fn slots_covered_by_pages(pages: usize) -> usize {
        let value_size = std::mem::size_of::<Slot<T>>();
        if value_size == 0 {
            usize::MAX
        } else {
            pages
                .checked_mul(vmem::page_size())
                .unwrap()
                .checked_div(value_size)
                .unwrap()
                .min(Self::max_index().saturating_add(1))
        }
    }

    #[inline(always)]
    fn is_committed_for_index(&self, index: usize) -> bool {
        debug_assert!(index <= Self::max_index());
        index < self.committed_slots.load(Ordering::Acquire)
    }

    #[inline(always)]
    fn decode_extra(index_and_lock: u32) -> Option<u32> {
        match index_and_lock {
            0 | 1 => None,
            _ => Some(index_and_lock - 2),
        }
    }

    fn ensure_committed_for_index(&self, index: usize) -> bool {
        let value_size = std::mem::size_of::<Slot<T>>();
        if value_size == 0 {
            return false;
        }
        let max_index = Self::max_index();
        assert!(index <= max_index);

        let min_pages = Self::pages_needed_for_index(index);
        let target_index = index.saturating_mul(2).min(max_index);
        let target_pages = Self::pages_needed_for_index(target_index);
        let max_ahead_pages = min_pages.saturating_add(LINEAR_COMMIT_AHEAD_PAGES);
        let desired_pages = target_pages.min(max_ahead_pages);

        if self.is_committed_for_index(target_index) {
            return false;
        }

        let mut committed_pages = self.committed_pages.lock();
        if desired_pages <= *committed_pages {
            return false;
        }

        let page_size = vmem::page_size();
        let commit_pages = desired_pages - *committed_pages;
        let commit_offset = (*committed_pages).checked_mul(page_size).unwrap();
        let commit_bytes = commit_pages.checked_mul(page_size).unwrap();
        if commit_bytes != 0 {
            let addr = unsafe { self.ptr.as_ptr().cast::<u8>().add(commit_offset) };
            vmem::commit(addr, commit_bytes)
                .unwrap_or_else(|err| panic!("failed to commit linear storage mapping: {err}"));
        }
        *committed_pages = desired_pages;
        self.committed_slots.store(Self::slots_covered_by_pages(desired_pages), Ordering::Release);
        true
    }

    /// SAFETY: `index` must be in bounds for this storage, and the caller must ensure the slot's
    /// backing pages are accessible for reads, which in practice means the slot was previously
    /// committed or `T` is zero-sized.
    #[inline(always)]
    unsafe fn get(&self, index: usize) -> Option<(&T, u32)> {
        let slot = self.slot_ptr(index);
        let extra = Self::decode_extra(unsafe { &(*slot).index_and_lock }.load(Ordering::Acquire))?;
        Some((unsafe { &(*slot).value }, extra))
    }

    /// SAFETY: `index` must be in bounds for this storage, and the caller must ensure the slot's
    /// backing pages are accessible for reads, which in practice means the slot was previously
    /// committed or `T` is zero-sized.
    #[inline(always)]
    unsafe fn get_copy(&self, index: usize) -> Option<(T, u32)>
    where
        T: Copy,
    {
        let slot = self.slot_ptr(index);
        let extra = Self::decode_extra(unsafe { &(*slot).index_and_lock }.load(Ordering::Acquire))?;
        Some((unsafe { (*slot).value }, extra))
    }

    #[inline(always)]
    fn put(&self, index: usize, value: T, extra: u32) -> bool {
        let _ = self.ensure_committed_for_index(index);
        let slot = self.slot_ptr(index);
        let index_and_lock = unsafe { &(*slot).index_and_lock };
        match index_and_lock.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                unsafe {
                    (&raw mut (*slot).value).write(value);
                }
                index_and_lock.store(extra.checked_add(2).unwrap(), Ordering::Release);
                true
            }
            Err(1) => panic!("caller raced calls to put()"),
            Err(_) => false,
        }
    }

    /// SAFETY: `index` must be in bounds for this storage, and the caller must guarantee this slot
    /// is uniquely initialized exactly once with no concurrent readers or writers observing it until
    /// after the final `index_and_lock.store(Ordering::Release)`.
    #[inline(always)]
    unsafe fn put_unique(&self, index: usize, value: T, extra: u32) {
        let _ = self.ensure_committed_for_index(index);
        let slot = self.slot_ptr(index);
        unsafe {
            (&raw mut (*slot).value).write(value);
        }
        let index_and_lock = unsafe { &(*slot).index_and_lock };
        index_and_lock.store(extra.checked_add(2).unwrap(), Ordering::Release);
    }
}

// SAFETY: No access to `T` is made.
unsafe impl<#[may_dangle] T> Drop for LinearStorage<T> {
    fn drop(&mut self) {
        assert!(!std::mem::needs_drop::<T>());

        let len_bytes = Self::mapping_len_bytes();
        if len_bytes != 0 {
            unsafe {
                vmem::release(self.ptr.as_ptr().cast::<u8>(), len_bytes).unwrap_or_else(|err| {
                    panic!("failed to release linear storage mapping: {err}")
                });
            }
        }
    }
}

unsafe impl<T: Send> Send for LinearStorage<T> {}
unsafe impl<T: Sync> Sync for LinearStorage<T> {}
unsafe impl<T: DynSend> DynSend for LinearStorage<T> {}
unsafe impl<T: DynSync> DynSync for LinearStorage<T> {}

struct LinearVec<T> {
    storage: LinearStorage<T>,
    len: AtomicUsize,
}

impl<T> LinearVec<T> {
    fn new() -> Self {
        LinearVec { storage: LinearStorage::new(), len: AtomicUsize::new(0) }
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    fn push(&self, value: T) -> usize {
        let index = self.len.fetch_add(1, Ordering::Relaxed);
        unsafe { self.storage.put_unique(index, value, 0) };
        index
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        let _ = self.storage.ensure_committed_for_index(index);
        unsafe { self.storage.get(index).map(|(value, _)| value) }
    }
}

// SAFETY: No access to `T` is made.
unsafe impl<#[may_dangle] T> Drop for LinearVec<T> {
    fn drop(&mut self) {
        assert!(!std::mem::needs_drop::<T>());
    }
}

pub struct LinearVecCache<K: Idx, V, I> {
    slots: LinearStorage<V>,
    present: LinearVec<(u32, u32)>,
    key: PhantomData<(K, I)>,
}

impl<K: Idx, V, I> Default for LinearVecCache<K, V, I> {
    fn default() -> Self {
        LinearVecCache { slots: LinearStorage::new(), present: LinearVec::new(), key: PhantomData }
    }
}

impl<K, V, I> LinearVecCache<K, V, I>
where
    K: Eq + Idx + Copy + Debug,
    V: Copy,
    I: Idx + Copy,
{
    #[inline(always)]
    pub fn lookup(&self, key: &K) -> Option<(V, I)> {
        let key_u32 = u32::try_from(key.index()).unwrap();
        let key_index = key_u32 as usize;
        if !self.slots.is_committed_for_index(key_index) {
            return None;
        }
        let (value, extra) = unsafe { self.slots.get_copy(key_index) }?;
        Some((value, I::new(extra as usize)))
    }

    #[inline]
    pub fn complete(&self, key: K, value: V, index: I) {
        let key_u32 = u32::try_from(key.index()).unwrap();
        let index_u32 = u32::try_from(index.index()).unwrap();
        if self.slots.put(key_u32 as usize, value, index_u32) {
            let _ = self.present.push((key_u32, index_u32));
        }
    }

    pub fn for_each(&self, f: &mut dyn FnMut(&K, &V, I)) {
        for idx in 0..self.present.len() {
            // `LinearVec::push` reserves the index before publishing the value, so iteration can
            // briefly observe the new length before the corresponding entry becomes readable.
            let Some((key_u32, extra)) = self.present.get(idx).copied() else {
                break;
            };
            let (value, _) =
                unsafe { self.slots.get(key_u32 as usize) }.expect("missing slot for present key");
            f(&K::new(key_u32 as usize), value, I::new(extra as usize));
        }
    }

    pub fn len(&self) -> usize {
        self.present.len()
    }
}
