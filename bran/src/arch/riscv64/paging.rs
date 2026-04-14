use kernel::{FrameAllocatorHook, MapKind, MapPerms};

#[derive(Clone, Copy, Default)]
pub struct RISCV64AddressSpace(pub u64);

static mut HHDM_OFFSET: u64 = 0;

pub fn init(offset: u64) {
    if offset == 0 {
        kernel::kprintln!("CRITICAL: paging::init called with offset 0");
    } else {
        kernel::kprintln!("paging::init setting HHDM_OFFSET = {:x}", offset);
    }
    unsafe { HHDM_OFFSET = offset };
}

pub fn active_address_space() -> RISCV64AddressSpace {
    let satp: u64;
    unsafe {
        core::arch::asm!("csrr {}, satp", out(reg) satp);
    }
    RISCV64AddressSpace(satp)
}

pub fn make_user_address_space(
    _active: RISCV64AddressSpace,
    allocator: &dyn FrameAllocatorHook,
) -> RISCV64AddressSpace {
    let phys = allocator.alloc_frame().expect("No frames for User SATP");
    let virt = phys + unsafe { HHDM_OFFSET };
    unsafe {
        core::ptr::write_bytes(virt as *mut u8, 0, 4096);
    }
    // Mode Sv39 = 8
    RISCV64AddressSpace((8 << 60) | (phys >> 12))
}

pub fn map_page(
    aspace: RISCV64AddressSpace,
    virt: u64,
    phys: u64,
    perms: MapPerms,
    _kind: MapKind,
    allocator: &dyn FrameAllocatorHook,
) -> Result<(), ()> {
    let mut bits = 1u64; // Valid
    if perms.read {
        bits |= 1 << 1;
    }
    if perms.write {
        bits |= 1 << 2;
    }
    if perms.exec {
        bits |= 1 << 3;
    }
    if perms.user {
        bits |= 1 << 4;
    }
    bits |= (1 << 6) | (1 << 7); // Accessed + Dirty

    // SATP format: [63:60]=Mode, [59:44]=ASID, [43:0]=PPN
    let mode = aspace.0 >> 60;

    // PPN * 4096 = physical address of root page table
    let root_phys = (aspace.0 & 0x0000_0FFF_FFFF_FFFF) << 12;
    let mut table = (root_phys + unsafe { HHDM_OFFSET }) as *mut u64;

    if mode == 9 {
        // Sv48 (4 levels)
        table = ensure_table(table, (virt >> 39) & 0x1ff, allocator)?;
    } else if mode != 8 { // Not Sv39 and not Sv48
        // Fallback or panic? For now assume Sv39 if not Sv48.
    }

    // Sv39 levels (3 levels) or continuation of Sv48
    let l2 = ensure_table(table, (virt >> 30) & 0x1ff, allocator)?;
    let l1 = ensure_table(l2, (virt >> 21) & 0x1ff, allocator)?;

    let pte_idx = (virt >> 12) & 0x1ff;
    // PTE format: [53:10]=PPN, [9:0]=flags
    let pte_val = ((phys >> 12) << 10) | bits;
    unsafe {
        *l1.add(pte_idx as usize) = pte_val;
    }

    tlb_flush_page(virt);
    Ok(())
}

fn ensure_table(
    parent: *mut u64,
    index: u64,
    allocator: &dyn FrameAllocatorHook,
) -> Result<*mut u64, ()> {
    let entry = unsafe { *parent.add(index as usize) };
    if entry & 1 == 0 {
        let phys = allocator.alloc_frame().ok_or(())?;
        unsafe {
            let virt = phys + HHDM_OFFSET;
            if HHDM_OFFSET == 0 {
                kernel::kprintln!("CRITICAL: HHDM_OFFSET is 0 in ensure_table!");
            }
            kernel::kprintln!("ensure_table: clearing virt={:x} (phys={:x})", virt, phys);
            core::ptr::write_bytes(virt as *mut u8, 0, 4096);
            // Non-leaf PTE: V=1, R=W=X=0, PPN set
            *parent.add(index as usize) = ((phys >> 12) << 10) | 1;
        }
        Ok((phys + unsafe { HHDM_OFFSET }) as *mut u64)
    } else {
        // Extract PPN from existing entry: bits [53:10]
        let ppn = (entry >> 10) & 0x00FF_FFFF_FFFF_FFFF;
        Ok(((ppn << 12) + unsafe { HHDM_OFFSET }) as *mut u64)
    }
}

pub fn unmap_page(_aspace: RISCV64AddressSpace, _virt: u64) -> Result<Option<u64>, ()> {
    Ok(None)
}
pub fn translate(_aspace: RISCV64AddressSpace, _virt: u64) -> Option<u64> {
    None
}
pub fn tlb_flush_page(_virt: u64) {
    unsafe {
        core::arch::asm!("sfence.vma");
    }
}
