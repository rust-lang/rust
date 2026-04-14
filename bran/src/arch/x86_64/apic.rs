// x86_64 Local APIC Access

use core::ptr;

pub fn base_phys() -> u64 {
    let eax: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") 0x1B_u32,
            out("eax") eax,
            out("edx") edx,
            options(nostack, preserves_flags)
        );
    }
    // Mask out reserved bits (low 12 bits) to get physical base address
    ((edx as u64) << 32 | eax as u64) & 0xFFFF_F000
}

pub fn id(base_phys: u64, hhdm: u64) -> u32 {
    let id_reg = base_phys + hhdm + 0x20;
    let val = unsafe { ptr::read_volatile(id_reg as *const u32) };
    val >> 24
}
