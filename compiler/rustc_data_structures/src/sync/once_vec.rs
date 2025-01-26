use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

/// Provides a singly-settable Vec.
///
/// This provides amortized, concurrent O(1) access to &T, expecting a densely numbered key space
/// (all value slots are allocated up to the highest key inserted).
pub struct OnceVec<T> {
    // Provide storage for up to 2^35 elements, which we expect to be enough in practice -- but can
    // be increased if needed. We may want to make the `slabs` list dynamic itself, likely by
    // indirecting through one more pointer to reduce space consumption of OnceVec if this grows
    // much larger.
    //
    // None of the code makes assumptions based on this size so bumping it up is easy.
    slabs: [Slab<T>; 36],
}

impl<T> Default for OnceVec<T> {
    fn default() -> Self {
        OnceVec { slabs: [const { Slab::new() }; 36] }
    }
}

unsafe impl<#[may_dangle] T> Drop for OnceVec<T> {
    fn drop(&mut self) {
        for (idx, slab) in self.slabs.iter_mut().enumerate() {
            unsafe { slab.deallocate(1 << idx) }
        }
    }
}

impl<T> OnceVec<T> {
    #[inline]
    fn to_slab_args(idx: usize) -> (usize, usize, usize) {
        let slab_idx = (idx + 1).ilog2() as usize;
        let cap = 1 << slab_idx;
        let idx_in_slab = idx - (cap - 1);
        (slab_idx, cap, idx_in_slab)
    }

    pub fn insert(&self, idx: usize, value: T) -> Result<(), T> {
        let (slab_idx, cap, idx_in_slab) = Self::to_slab_args(idx);
        self.slabs[slab_idx].insert(cap, idx_in_slab, value)
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        let (slab_idx, cap, idx_in_slab) = Self::to_slab_args(idx);
        self.slabs[slab_idx].get(cap, idx_in_slab)
    }
}

struct Slab<T> {
    // If non-zero, points to a contiguously allocated block which starts with a bitset
    // (two bits per value, one for whether a value is present and the other for whether a value is
    // currently being written) and then `[V]` (some of which may be missing).
    //
    // The capacity is implicit and passed with all accessors.
    v: AtomicPtr<u8>,
    _phantom: PhantomData<[T; 1]>,
}

impl<T> Slab<T> {
    const fn new() -> Slab<T> {
        Slab { v: AtomicPtr::new(std::ptr::null_mut()), _phantom: PhantomData }
    }

    fn initialize(&self, cap: usize) -> NonNull<u8> {
        static LOCK: Mutex<()> = Mutex::new(());

        if let Some(ptr) = NonNull::new(self.v.load(Ordering::Acquire)) {
            return ptr;
        }

        // If we are initializing the bucket, then acquire a global lock.
        //
        // This path is quite cold, so it's cheap to use a global lock. This ensures that we never
        // have multiple allocations for the same bucket.
        let _allocator_guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Check the lock again, sicne we might have been initialized while waiting on the lock.
        if let Some(ptr) = NonNull::new(self.v.load(Ordering::Acquire)) {
            return ptr;
        }

        let layout = Self::layout(cap).0;
        assert!(layout.size() > 0);

        // SAFETY: Checked above that layout is non-zero sized.
        let Some(allocation) = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }) else {
            std::alloc::handle_alloc_error(layout);
        };

        self.v.store(allocation.as_ptr(), Ordering::Release);

        allocation
    }

    fn bitset(ptr: NonNull<u8>, cap: usize) -> NonNull<[AtomicU8]> {
        NonNull::slice_from_raw_parts(ptr.cast(), cap.div_ceil(4))
    }

    // SAFETY: Must be called on a `initialize`d `ptr` for this capacity.
    unsafe fn slice(ptr: NonNull<u8>, cap: usize) -> NonNull<[MaybeUninit<T>]> {
        let offset = Self::layout(cap).1;
        // SAFETY: Passed up to caller.
        NonNull::slice_from_raw_parts(unsafe { ptr.add(offset).cast() }, cap)
    }

    // idx is already compacted to within this slab
    fn get(&self, cap: usize, idx: usize) -> Option<&T> {
        // avoid initializing for get queries
        let Some(ptr) = NonNull::new(self.v.load(Ordering::Acquire)) else {
            return None;
        };

        let bitset = unsafe { Self::bitset(ptr, cap).as_ref() };

        // Check if the entry is initialized.
        //
        // Bottom 4 bits are the "is initialized" bits, top 4 bits are used for "is initializing"
        // lock.
        let word = bitset[idx / 4].load(Ordering::Acquire);
        if word & (1 << (idx % 4)) == 0 {
            return None;
        }

        // Avoid as_ref() since we don't want to assert shared refs to all slots (some are being
        // concurrently updated).
        //
        // SAFETY: `ptr` is only written by `initialize`, so this is safe.
        let slice = unsafe { Self::slice(ptr, cap) };
        assert!(idx < slice.len());
        // SAFETY: assertion above checks that we're in-bounds.
        let slot = unsafe { slice.cast::<T>().add(idx) };

        // SAFETY: We checked `bitset` and this value was initialized. Our Acquire load
        // establishes the memory ordering with the release store which set the bit, so we're safe
        // to read it.
        Some(unsafe { slot.as_ref() })
    }

    // idx is already compacted to within this slab
    fn insert(&self, cap: usize, idx: usize, value: T) -> Result<(), T> {
        // avoid initializing for get queries
        let ptr = self.initialize(cap);
        let bitset = unsafe { Self::bitset(ptr, cap).as_ref() };

        // Check if the entry is initialized, and lock it for writing.
        let word = bitset[idx / 4].fetch_or(1 << (4 + idx % 4), Ordering::AcqRel);
        if word & (1 << (idx % 4)) != 0 {
            // Already fully initialized prior to us setting the "is writing" bit.
            return Err(value);
        }
        if word & (1 << (4 + idx % 4)) != 0 {
            // Someone else already acquired the lock for writing.
            return Err(value);
        }

        let slice = unsafe { Self::slice(ptr, cap) };
        assert!(idx < slice.len());
        // SAFETY: assertion above checks that we're in-bounds.
        let slot = unsafe { slice.cast::<T>().add(idx) };

        // SAFETY: We locked this slot for writing with the fetch_or above, and were the first to do
        // so (checked in 2nd `if` above).
        unsafe {
            slot.write(value);
        }

        // Set the is-present bit, indicating that we have finished writing this value.
        // Acquire ensures we don't break synchronizes-with relationships in other bits (unclear if
        // strictly necessary but definitely doesn't hurt).
        bitset[idx / 4].fetch_or(1 << (idx % 4), Ordering::AcqRel);

        Ok(())
    }

    /// Returns the layout for a Slab with capacity for `cap` elements, and the offset into the
    /// allocation at which the T slice starts.
    fn layout(cap: usize) -> (Layout, usize) {
        Layout::array::<AtomicU8>(cap.div_ceil(4))
            .unwrap()
            .extend(Layout::array::<T>(cap).unwrap())
            .unwrap()
    }

    // Drop, except passing the capacity
    unsafe fn deallocate(&mut self, cap: usize) {
        // avoid initializing just to Drop
        let Some(ptr) = NonNull::new(self.v.load(Ordering::Acquire)) else {
            return;
        };

        if std::mem::needs_drop::<T>() {
            // SAFETY: `ptr` is only written by `initialize`, and zero-init'd so AtomicU8 is present in
            // the bitset range.
            let bitset = unsafe { Self::bitset(ptr, cap).as_ref() };
            // SAFETY: `ptr` is only written by `initialize`, so satisfies slice precondition.
            let slice = unsafe { Self::slice(ptr, cap).cast::<T>() };

            for (word_idx, word) in bitset.iter().enumerate() {
                let word = word.load(Ordering::Acquire);
                for word_offset in 0..4 {
                    if word & (1 << word_offset) != 0 {
                        // Was initialized, need to drop the value.
                        let idx = word_idx * 4 + word_offset;
                        unsafe {
                            std::ptr::drop_in_place(slice.add(idx).as_ptr());
                        }
                    }
                }
            }
        }

        let layout = Self::layout(cap).0;

        // SAFETY: Allocated with `alloc` and the same layout.
        unsafe {
            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod test;
