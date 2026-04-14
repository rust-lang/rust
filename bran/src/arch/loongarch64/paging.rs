use kernel::{FrameAllocatorHook, MapKind, MapPerms};

#[derive(Clone, Copy, Default)]
pub struct LoongArch64AddressSpace {
    pub pgdl: u64,
    pub pgdh: u64,
}

static mut HHDM_OFFSET: u64 = 0;

pub fn init(offset: u64) {
    unsafe { HHDM_OFFSET = offset };
}

pub fn active_address_space() -> LoongArch64AddressSpace {
    let pgdl: u64;
    let pgdh: u64;
    unsafe {
        core::arch::asm!("csrrd {}, 0x19", out(reg) pgdl); // PGDL
        core::arch::asm!("csrrd {}, 0x1a", out(reg) pgdh); // PGDH
    }
    LoongArch64AddressSpace { pgdl, pgdh }
}

pub fn make_user_address_space(
    active: LoongArch64AddressSpace,
    allocator: &dyn FrameAllocatorHook,
) -> LoongArch64AddressSpace {
    let pgdl = allocator.alloc_frame().expect("No frames for User AS");
    let virt = pgdl + unsafe { HHDM_OFFSET };
    unsafe {
        core::ptr::write_bytes(virt as *mut u8, 0, 4096);
    }
    LoongArch64AddressSpace {
        pgdl,
        pgdh: active.pgdh,
    }
}

pub fn map_page(
    aspace: LoongArch64AddressSpace,
    virt: u64,
    phys: u64,
    perms: MapPerms,
    kind: MapKind,
    allocator: &dyn FrameAllocatorHook,
) -> Result<(), ()> {
    let mut bits = 1u64 << 0; // V (Valid)
    if perms.write {
        bits |= 1 << 1;
    } // D (Dirty/Writable)

    // PLV (Privilege Level): Kernel=0, User=3
    if perms.user {
        bits |= 3 << 2;
    } // PLV=3 (Set bits 2 and 3)

    // MAT (Memory Access Type): Normal=1 (CC), Device=0 (SU)
    if kind == MapKind::Normal {
        bits |= 1 << 4;
    } // MAT=1 (Coherent Cached)

    if !perms.exec {
        bits |= 1 << 62;
    } // NX

    let root_phys = if (virt >> 63) != 0 {
        aspace.pgdh
    } else {
        aspace.pgdl
    };
    let root = (root_phys + unsafe { HHDM_OFFSET }) as *mut u64;
    // LoongArch64 uses 4-level page tables (48-bit VA): L3 -> L2 -> L1 -> L0.
    let l3 = ensure_table(root, (virt >> 39) & 0x1ff, allocator)?;
    let l2 = ensure_table(l3, (virt >> 30) & 0x1ff, allocator)?;
    let l1 = ensure_table(l2, (virt >> 21) & 0x1ff, allocator)?;

    let pte_idx = (virt >> 12) & 0x1ff;
    unsafe {
        *l1.add(pte_idx as usize) = (phys & !0xfff) | bits;
    }
    Ok(())
}

fn ensure_table(
    parent: *mut u64,
    index: u64,
    allocator: &dyn FrameAllocatorHook,
) -> Result<*mut u64, ()> {
    let entry = unsafe { *parent.add(index as usize) };
    if entry == 0 {
        let phys = allocator.alloc_frame().ok_or(())?;
        unsafe {
            let virt = phys + HHDM_OFFSET;
            core::ptr::write_bytes(virt as *mut u8, 0, 4096);
            *parent.add(index as usize) = phys | 1 | (1 << 1) | (1 << 4); // Valid(0) | Dirty(1) | MAT=CC(4)
        }
        Ok((phys + unsafe { HHDM_OFFSET }) as *mut u64)
    } else {
        Ok(((entry & !0xFFF) + unsafe { HHDM_OFFSET }) as *mut u64)
    }
}

pub fn unmap_page(_aspace: LoongArch64AddressSpace, _virt: u64) -> Result<Option<u64>, ()> {
    Ok(None)
}
pub fn translate(_aspace: LoongArch64AddressSpace, _virt: u64) -> Option<u64> {
    None
}
pub fn tlb_flush_page(virt: u64) {
    unsafe {
        core::arch::asm!("invtlb 0x7, $zero, {}", in(reg) virt);
    }
}
