//! Manages a pool of addresses that can be reused.

use rand::Rng;

use rustc_target::abi::{Align, Size};

const MAX_POOL_SIZE: usize = 64;

// Just use fair coins, until we have evidence that other numbers are better.
const ADDR_REMEMBER_CHANCE: f64 = 0.5;
const ADDR_TAKE_CHANCE: f64 = 0.5;

/// The pool strikes a balance between exploring more possible executions and making it more likely
/// to find bugs. The hypothesis is that bugs are more likely to occur when reuse happens for
/// allocations with the same layout, since that can trigger e.g. ABA issues in a concurrent data
/// structure. Therefore we only reuse allocations when size and alignment match exactly.
#[derive(Debug)]
pub struct ReusePool {
    /// The i-th element in `pool` stores allocations of alignment `2^i`. We store these reusable
    /// allocations as address-size pairs, the list must be sorted by the size.
    ///
    /// Each of these maps has at most MAX_POOL_SIZE elements, and since alignment is limited to
    /// less than 64 different possible value, that bounds the overall size of the pool.
    pool: Vec<Vec<(u64, Size)>>,
}

impl ReusePool {
    pub fn new() -> Self {
        ReusePool { pool: vec![] }
    }

    fn subpool(&mut self, align: Align) -> &mut Vec<(u64, Size)> {
        let pool_idx: usize = align.bytes().trailing_zeros().try_into().unwrap();
        if self.pool.len() <= pool_idx {
            self.pool.resize(pool_idx + 1, Vec::new());
        }
        &mut self.pool[pool_idx]
    }

    pub fn add_addr(&mut self, rng: &mut impl Rng, addr: u64, size: Size, align: Align) {
        // Let's see if we even want to remember this address.
        if !rng.gen_bool(ADDR_REMEMBER_CHANCE) {
            return;
        }
        // Determine the pool to add this to, and where in the pool to put it.
        let subpool = self.subpool(align);
        let pos = subpool.partition_point(|(_addr, other_size)| *other_size < size);
        // Make sure the pool does not grow too big.
        if subpool.len() >= MAX_POOL_SIZE {
            // Pool full. Replace existing element, or last one if this would be even bigger.
            let clamped_pos = pos.min(subpool.len() - 1);
            subpool[clamped_pos] = (addr, size);
            return;
        }
        // Add address to pool, at the right position.
        subpool.insert(pos, (addr, size));
    }

    pub fn take_addr(&mut self, rng: &mut impl Rng, size: Size, align: Align) -> Option<u64> {
        // Determine whether we'll even attempt a reuse.
        if !rng.gen_bool(ADDR_TAKE_CHANCE) {
            return None;
        }
        // Determine the pool to take this from.
        let subpool = self.subpool(align);
        // Let's see if we can find something of the right size. We want to find the full range of
        // such items, beginning with the first, so we can't use `binary_search_by_key`.
        let begin = subpool.partition_point(|(_addr, other_size)| *other_size < size);
        let mut end = begin;
        while let Some((_addr, other_size)) = subpool.get(end) {
            if *other_size != size {
                break;
            }
            end += 1;
        }
        if end == begin {
            // Could not find any item of the right size.
            return None;
        }
        // Pick a random element with the desired size.
        let idx = rng.gen_range(begin..end);
        // Remove it from the pool and return.
        let (chosen_addr, chosen_size) = subpool.remove(idx);
        debug_assert!(chosen_size >= size && chosen_addr % align.bytes() == 0);
        Some(chosen_addr)
    }
}
