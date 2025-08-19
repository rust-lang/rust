//! Manages a pool of addresses that can be reused.

use rand::Rng;
use rustc_abi::{Align, Size};

use crate::concurrency::VClock;
use crate::helpers::ToUsize as _;
use crate::{MemoryKind, MiriConfig, ThreadId};

const MAX_POOL_SIZE: usize = 64;

/// The pool strikes a balance between exploring more possible executions and making it more likely
/// to find bugs. The hypothesis is that bugs are more likely to occur when reuse happens for
/// allocations with the same layout, since that can trigger e.g. ABA issues in a concurrent data
/// structure. Therefore we only reuse allocations when size and alignment match exactly.
#[derive(Debug)]
pub struct ReusePool {
    address_reuse_rate: f64,
    address_reuse_cross_thread_rate: f64,
    /// The i-th element in `pool` stores allocations of alignment `2^i`. We store these reusable
    /// allocations as address-size pairs, the list must be sorted by the size and then the thread ID.
    ///
    /// Each of these maps has at most MAX_POOL_SIZE elements, and since alignment is limited to
    /// less than 64 different possible values, that bounds the overall size of the pool.
    ///
    /// We also store the ID and the data-race clock of the thread that donated this pool element,
    /// to ensure synchronization with the thread that picks up this address.
    pool: Vec<Vec<(u64, Size, ThreadId, VClock)>>,
}

impl ReusePool {
    pub fn new(config: &MiriConfig) -> Self {
        ReusePool {
            address_reuse_rate: config.address_reuse_rate,
            address_reuse_cross_thread_rate: config.address_reuse_cross_thread_rate,
            pool: vec![],
        }
    }

    /// Call this when we are using up a lot of the address space: if memory reuse is enabled at all,
    /// this will bump the intra-thread reuse rate to 100% so that we can keep running this program as
    /// long as possible.
    pub fn address_space_shortage(&mut self) {
        if self.address_reuse_rate > 0.0 {
            self.address_reuse_rate = 1.0;
        }
    }

    fn subpool(&mut self, align: Align) -> &mut Vec<(u64, Size, ThreadId, VClock)> {
        let pool_idx: usize = align.bytes().trailing_zeros().to_usize();
        if self.pool.len() <= pool_idx {
            self.pool.resize(pool_idx + 1, Vec::new());
        }
        &mut self.pool[pool_idx]
    }

    pub fn add_addr(
        &mut self,
        rng: &mut impl Rng,
        addr: u64,
        size: Size,
        align: Align,
        kind: MemoryKind,
        thread: ThreadId,
        clock: impl FnOnce() -> VClock,
    ) {
        // Let's see if we even want to remember this address.
        // We don't remember stack addresses since there's so many of them (so the perf impact is big).
        if kind == MemoryKind::Stack || !rng.random_bool(self.address_reuse_rate) {
            return;
        }
        let clock = clock();
        // Determine the pool to add this to, and where in the pool to put it.
        let subpool = self.subpool(align);
        let pos = subpool.partition_point(|(_addr, other_size, other_thread, _)| {
            (*other_size, *other_thread) < (size, thread)
        });
        // Make sure the pool does not grow too big.
        if subpool.len() >= MAX_POOL_SIZE {
            // Pool full. Replace existing element, or last one if this would be even bigger.
            let clamped_pos = pos.min(subpool.len() - 1);
            subpool[clamped_pos] = (addr, size, thread, clock);
            return;
        }
        // Add address to pool, at the right position.
        subpool.insert(pos, (addr, size, thread, clock));
    }

    /// Returns the address to use and optionally a clock we have to synchronize with.
    pub fn take_addr(
        &mut self,
        rng: &mut impl Rng,
        size: Size,
        align: Align,
        kind: MemoryKind,
        thread: ThreadId,
    ) -> Option<(u64, Option<VClock>)> {
        // Determine whether we'll even attempt a reuse. As above, we don't do reuse for stack addresses.
        if kind == MemoryKind::Stack || !rng.random_bool(self.address_reuse_rate) {
            return None;
        }
        let cross_thread_reuse = rng.random_bool(self.address_reuse_cross_thread_rate);
        // Determine the pool to take this from.
        let subpool = self.subpool(align);
        // Let's see if we can find something of the right size. We want to find the full range of
        // such items, beginning with the first, so we can't use `binary_search_by_key`. If we do
        // *not* want to consider other thread's allocations, we effectively use the lexicographic
        // order on `(size, thread)`.
        let begin = subpool.partition_point(|(_addr, other_size, other_thread, _)| {
            *other_size < size
                || (*other_size == size && !cross_thread_reuse && *other_thread < thread)
        });
        let mut end = begin;
        while let Some((_addr, other_size, other_thread, _)) = subpool.get(end) {
            if *other_size != size {
                break;
            }
            if !cross_thread_reuse && *other_thread != thread {
                // We entered the allocations of another thread.
                break;
            }
            end += 1;
        }
        if end == begin {
            // Could not find any item of the right size.
            return None;
        }
        // Pick a random element with the desired size.
        let idx = rng.random_range(begin..end);
        // Remove it from the pool and return.
        let (chosen_addr, chosen_size, chosen_thread, clock) = subpool.remove(idx);
        debug_assert!(chosen_size >= size && chosen_addr.is_multiple_of(align.bytes()));
        debug_assert!(cross_thread_reuse || chosen_thread == thread);
        // No synchronization needed if we reused from the current thread.
        Some((chosen_addr, if chosen_thread == thread { None } else { Some(clock) }))
    }
}
