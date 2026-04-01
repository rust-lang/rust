use std::ops::Range;

use rand::Rng;
use rustc_abi::{Align, Size};
use rustc_const_eval::interpret::{InterpResult, interp_ok};
use rustc_middle::{err_exhaust, throw_exhaust};

/// Shifts `addr` to make it aligned with `align` by rounding `addr` to the smallest multiple
/// of `align` that is larger or equal to `addr`
fn align_addr(addr: u64, align: u64) -> u64 {
    match addr % align {
        0 => addr,
        rem => addr.strict_add(align) - rem,
    }
}

/// This provides the logic to generate addresses for memory allocations in a given address range.
#[derive(Debug)]
pub struct AddressGenerator {
    /// This is used as a memory address when a new pointer is casted to an integer. It
    /// is always larger than any address that was previously made part of a block.
    next_base_addr: u64,
    /// This is the last address that can be allocated.
    end: u64,
}

impl AddressGenerator {
    pub fn new(addr_range: Range<u64>) -> Self {
        Self { next_base_addr: addr_range.start, end: addr_range.end }
    }

    /// Get the remaining range where this `AddressGenerator` can still allocate addresses.
    pub fn get_remaining(&self) -> Range<u64> {
        self.next_base_addr..self.end
    }

    /// Generate a new address with the specified size and alignment, using the given Rng to add some randomness.
    /// The returned allocation is guaranteed not to overlap with any address ranges given out by the generator before.
    /// Returns an error if the allocation request cannot be fulfilled.
    pub fn generate<'tcx, R: Rng>(
        &mut self,
        size: Size,
        align: Align,
        rng: &mut R,
    ) -> InterpResult<'tcx, u64> {
        // Leave some space to the previous allocation, to give it some chance to be less aligned.
        // We ensure that `(self.next_base_addr + slack) % 16` is uniformly distributed.
        let slack = rng.random_range(0..16);
        // From next_base_addr + slack, round up to adjust for alignment.
        let base_addr =
            self.next_base_addr.checked_add(slack).ok_or_else(|| err_exhaust!(AddressSpaceFull))?;
        let base_addr = align_addr(base_addr, align.bytes());

        // Remember next base address.  If this allocation is zero-sized, leave a gap of at
        // least 1 to avoid two allocations having the same base address. (The logic in
        // `alloc_id_from_addr` assumes unique addresses, and different function/vtable pointers
        // need to be distinguishable!)
        self.next_base_addr = base_addr
            .checked_add(size.bytes().max(1))
            .ok_or_else(|| err_exhaust!(AddressSpaceFull))?;
        // Even if `Size` didn't overflow, we might still have filled up the address space.
        if self.next_base_addr > self.end {
            throw_exhaust!(AddressSpaceFull);
        }
        interp_ok(base_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_addr() {
        assert_eq!(align_addr(37, 4), 40);
        assert_eq!(align_addr(44, 4), 44);
    }
}
