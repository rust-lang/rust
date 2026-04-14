use kernel::{FrameAllocatorHook, MapKind, MapPerms};

#[derive(Clone, Copy, Default)]
pub struct AArch64AddressSpace(pub u64);

static mut HHDM_OFFSET: u64 = 0;

pub fn init(offset: u64) {
    unsafe { HHDM_OFFSET = offset };
}

pub fn active_address_space() -> AArch64AddressSpace {
    // For kernel use, we need TTBR1 (kernel address space)
    let ttbr1: u64;
    unsafe {
        core::arch::asm!("mrs {}, ttbr1_el1", out(reg) ttbr1);
    }
    AArch64AddressSpace(ttbr1 & 0x0000_FFFF_FFFF_F000)
}

pub fn make_user_address_space(
    _active: AArch64AddressSpace,
    allocator: &dyn FrameAllocatorHook,
) -> AArch64AddressSpace {
    let phys = allocator.alloc_frame().expect("No frames for User TTBR0");
    let virt = phys + unsafe { HHDM_OFFSET };
    let ptr = virt as *mut u64;

    unsafe {
        core::ptr::write_bytes(ptr as *mut u8, 0, 4096);
    }

    AArch64AddressSpace(phys)
}

pub fn map_page(
    aspace: AArch64AddressSpace,
    virt: u64,
    phys: u64,
    perms: MapPerms,
    kind: MapKind,
    allocator: &dyn FrameAllocatorHook,
) -> Result<(), ()> {
    // AArch64 4-level, 4KB pages (48-bit VA)
    // MAIR index: 0 = Normal WB, 1 = Device nGnRE (set in MAIR_EL1)
    let mut attr_idx = 0u64;
    if kind == MapKind::Device {
        attr_idx = 1;
    }

    // Page descriptor bits:
    // [1:0] = 0b11 (valid page)
    // [4:2] = AttrIndx
    // [6]   = AP[1] (0=EL1 only, 1=EL0 accessible)
    // [7]   = AP[2] (0=RW, 1=RO)
    // [10]  = AF (Access Flag)
    // [53]  = PXN (Privileged Execute Never)
    // [54]  = UXN (User Execute Never)
    let mut desc = (phys & 0x0000_FFFF_FFFF_F000) | 0x3 | (attr_idx << 2) | (1 << 10);

    if !perms.write {
        desc |= 1 << 7;
    }
    if perms.user {
        desc |= 1 << 6;
    } // EL0 accessible
    if !perms.exec {
        desc |= (1 << 54) | (1 << 53);
    }

    let root = (aspace.0 + unsafe { HHDM_OFFSET }) as *mut u64;
    let l1 = ensure_table(root, (virt >> 39) & 0x1ff, allocator)?;
    let l2 = ensure_table(l1, (virt >> 30) & 0x1ff, allocator)?;
    let l3 = ensure_table(l2, (virt >> 21) & 0x1ff, allocator)?;

    let l3_idx = (virt >> 12) & 0x1ff;
    unsafe {
        *l3.add(l3_idx as usize) = desc;
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
            core::ptr::write_bytes(virt as *mut u8, 0, 4096);
            *parent.add(index as usize) = phys | 0x3; // Table descriptor
        }
        Ok((phys + unsafe { HHDM_OFFSET }) as *mut u64)
    } else {
        Ok(((entry & 0x0000_FFFF_FFFF_F000) + unsafe { HHDM_OFFSET }) as *mut u64)
    }
}

pub fn unmap_page(_aspace: AArch64AddressSpace, _virt: u64) -> Result<Option<u64>, ()> {
    Ok(None)
}
pub fn translate(_aspace: AArch64AddressSpace, _virt: u64) -> Option<u64> {
    None
}
pub fn tlb_flush_page(virt: u64) {
    unsafe {
        core::arch::asm!("tlbi vaae1is, {}", in(reg) virt >> 12);
        core::arch::asm!("dsb ish", "isb");
    }
}
