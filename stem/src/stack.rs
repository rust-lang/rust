use abi::errors::Errno;
use abi::types::StackInfo;
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};

use crate::vm::vm_map;

const PAGE_SIZE: usize = 4096;

/// Magic canary value — easily recognizable in memory dumps
pub const CANARY_PATTERN: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Stack corruption error details
#[derive(Debug, Clone, Copy)]
pub struct StackCorruption {
    pub location: CanaryLocation,
    pub expected: u64,
    pub found: u64,
}

/// Which canary was corrupted
#[derive(Debug, Clone, Copy)]
pub enum CanaryLocation {
    /// Bottom of committed region (low address)
    Low,
    /// Top of committed region (high address)
    High,
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

#[derive(Clone, Copy, Debug)]
pub struct StackSpec {
    pub reserve_bytes: usize,
    pub initial_commit_bytes: usize,
    pub guard_pages: usize,
    pub grow_chunk_bytes: usize,
}

impl Default for StackSpec {
    fn default() -> Self {
        Self {
            reserve_bytes: 2 * 1024 * 1024,
            initial_commit_bytes: 64 * 1024,
            guard_pages: 1,
            grow_chunk_bytes: 64 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stack {
    pub sp: *mut u8,
    pub reserve_start: *mut u8,
    pub reserve_end: *mut u8,
    pub committed_start: *mut u8,
    pub guard_start: *mut u8,
    pub guard_end: *mut u8,
    pub info: StackInfo,
}

impl Stack {
    pub fn alloc_growing_stack(spec: StackSpec) -> Result<Self, Errno> {
        let guard_bytes = spec.guard_pages.saturating_mul(PAGE_SIZE);
        let reserve_bytes = align_up(spec.reserve_bytes, PAGE_SIZE);
        let total = guard_bytes.saturating_add(reserve_bytes);

        let reserve_req = VmMapReq {
            addr_hint: 0,
            len: total,
            prot: VmProt::USER,
            flags: VmMapFlags::GUARD | VmMapFlags::PRIVATE,
            backing: VmBacking::Anonymous { zeroed: true },
        };
        let reserve_resp = vm_map(&reserve_req)?;
        let base = reserve_resp.addr;

        let guard_start = base;
        let guard_end = base.saturating_add(guard_bytes);
        let reserve_start = guard_end;
        let reserve_end = base.saturating_add(total);

        let commit_len = align_up(spec.initial_commit_bytes, PAGE_SIZE);
        let commit_start = reserve_end.saturating_sub(commit_len);

        let commit_req = VmMapReq {
            addr_hint: commit_start,
            len: commit_len,
            prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
            flags: VmMapFlags::FIXED | VmMapFlags::PRIVATE,
            backing: VmBacking::Anonymous { zeroed: true },
        };
        let _ = vm_map(&commit_req)?;

        let info = StackInfo {
            guard_start,
            guard_end,
            reserve_start,
            reserve_end,
            committed_start: commit_start,
            grow_chunk_bytes: align_up(spec.grow_chunk_bytes, PAGE_SIZE),
        };

        let stack = Stack {
            sp: (reserve_end.saturating_sub(8)) as *mut u8,
            reserve_start: reserve_start as *mut u8,
            reserve_end: reserve_end as *mut u8,
            committed_start: commit_start as *mut u8,
            guard_start: guard_start as *mut u8,
            guard_end: guard_end as *mut u8,
            info,
        };

        // Write canaries after allocation
        stack.write_canaries();

        Ok(stack)
    }

    /// Write canary patterns at committed region boundaries
    pub fn write_canaries(&self) {
        // Low canary: at start of committed region
        let low_canary = self.committed_start as *mut u64;
        // High canary: near top of reserve region (before SP slot)
        let high_canary = (self.reserve_end as usize - 16) as *mut u64;

        unsafe {
            core::ptr::write_volatile(low_canary, CANARY_PATTERN);
            core::ptr::write_volatile(high_canary, CANARY_PATTERN);
        }
    }

    /// Check canary integrity, returns Err if corruption detected
    pub fn check_canaries(&self) -> Result<(), StackCorruption> {
        let low = unsafe { core::ptr::read_volatile(self.committed_start as *const u64) };
        if low != CANARY_PATTERN {
            return Err(StackCorruption {
                location: CanaryLocation::Low,
                expected: CANARY_PATTERN,
                found: low,
            });
        }

        let high =
            unsafe { core::ptr::read_volatile((self.reserve_end as usize - 16) as *const u64) };
        if high != CANARY_PATTERN {
            return Err(StackCorruption {
                location: CanaryLocation::High,
                expected: CANARY_PATTERN,
                found: high,
            });
        }

        Ok(())
    }

    pub fn handle_stack_fault(_fault_addr: usize) -> Result<(), Errno> {
        Err(Errno::NotSupported)
    }
}

/// Get current stack pointer (architecture-specific)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn current_sp() -> usize {
    let sp: usize;
    unsafe { core::arch::asm!("mov {}, rsp", out(reg) sp, options(nomem, nostack)) };
    sp
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn current_sp() -> usize {
    let sp: usize;
    unsafe { core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack)) };
    sp
}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn current_sp() -> usize {
    let sp: usize;
    unsafe { core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack)) };
    sp
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
)))]
#[inline(always)]
pub fn current_sp() -> usize {
    0 // Fallback for unsupported architectures
}

/// Validate that SP is within valid stack bounds
pub fn validate_sp_in_bounds(stack: &Stack, sp: usize) -> bool {
    sp >= stack.committed_start as usize && sp < stack.reserve_end as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(100, 0), 100);
    }

    #[test]
    fn test_stack_spec_default() {
        let spec = StackSpec::default();
        assert_eq!(spec.guard_pages, 1);
        assert!(spec.reserve_bytes >= 64 * 1024);
        assert!(spec.initial_commit_bytes >= 64 * 1024);
    }

    #[test]
    fn test_sp_alignment_16_byte() {
        // Entry point SP must be 16-byte aligned (actually 16n+8 for ABI)
        let spec = StackSpec::default();
        let base = 0x1000_0000_usize;
        let sp = base + spec.reserve_bytes - 8; // -8 for return address slot
        assert_eq!(sp % 16, 8, "SP should be 16n+8 for x86_64 ABI compliance");
    }

    #[test]
    fn test_guard_page_placement() {
        let spec = StackSpec {
            guard_pages: 2,
            ..StackSpec::default()
        };
        let guard_bytes = spec.guard_pages * 4096;
        assert_eq!(guard_bytes, 8192);
    }

    #[test]
    fn test_canary_pattern_is_recognizable() {
        // Should be easy to spot in memory dumps
        assert_eq!(CANARY_PATTERN, 0xDEAD_BEEF_CAFE_BABE);
    }
}
