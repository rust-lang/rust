use core::sync::atomic::Ordering;
use kernel::CpuId;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "loongarch64")]
pub mod loongarch64;
#[cfg(target_arch = "riscv64")]
pub mod riscv64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub use aarch64::hcf;
#[cfg(target_arch = "loongarch64")]
pub use loongarch64::hcf;
#[cfg(target_arch = "riscv64")]
pub use riscv64::hcf;
#[cfg(target_arch = "x86_64")]
pub use x86_64::hcf;

#[cfg(target_arch = "x86_64")]
pub type CurrentRuntime = crate::runtime::Runtime<x86_64::X86_64Runtime>;
#[cfg(target_arch = "aarch64")]
pub type CurrentRuntime = crate::runtime::Runtime<aarch64::AArch64Runtime>;
#[cfg(target_arch = "riscv64")]
pub type CurrentRuntime = crate::runtime::Runtime<riscv64::RISCV64Runtime>;
#[cfg(target_arch = "loongarch64")]
pub type CurrentRuntime = crate::runtime::Runtime<loongarch64::LoongArch64Runtime>;

pub const fn create_runtime() -> CurrentRuntime {
    crate::runtime::Runtime {
        #[cfg(target_arch = "x86_64")]
        arch: x86_64::X86_64Runtime::new(),
        #[cfg(target_arch = "aarch64")]
        arch: aarch64::AArch64Runtime::new(),
        #[cfg(target_arch = "riscv64")]
        arch: riscv64::RISCV64Runtime::new(),
        #[cfg(target_arch = "loongarch64")]
        arch: loongarch64::LoongArch64Runtime::new(),
        limine: crate::runtime::LimineRuntimeData::new(),
    }
}

pub fn init_paging() {
    let offset = crate::requests::HHDM_REQUEST
        .get_response()
        .map(|r| r.offset())
        .unwrap_or(0);
    #[cfg(target_arch = "x86_64")]
    x86_64::paging::init(offset);
    #[cfg(target_arch = "aarch64")]
    aarch64::paging::init(offset);
    #[cfg(target_arch = "riscv64")]
    riscv64::paging::init(offset);
    #[cfg(target_arch = "loongarch64")]
    loongarch64::paging::init(offset);
}

pub unsafe fn init_interrupts() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::vector::init();
    }
}

/// Initialize IOAPIC for x86_64
#[cfg(target_arch = "x86_64")]
pub fn init_ioapic() {
    init_x86_64_ioapic();
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn init_ioapic() {}

#[cfg(target_arch = "x86_64")]
fn init_x86_64_ioapic() {
    use kernel::kdebug;

    let hhdm = crate::requests::HHDM_REQUEST
        .get_response()
        .map(|r| r.offset())
        .unwrap_or(0);

    kernel::kdebug!("IOAPIC: hhdm=0x{:x}", hhdm);

    // 1. Disable the legacy PIC
    kernel::kdebug!("IOAPIC: Disabling legacy PIC...");
    x86_64::pic::disable_pic();
    kernel::kdebug!("IOAPIC: PIC disabled OK");

    // 2. Parse ACPI MADT to find IOAPIC
    // Limine (Base Revision 4+) returns HHDM-mapped virtual address for RSDP
    let rsdp_virt = match crate::requests::RSDP_REQUEST.get_response() {
        Some(r) => r.address() as u64,
        None => {
            kernel::kdebug!("IOAPIC: No ACPI RSDP available, skipping");
            return;
        }
    };
    kernel::kdebug!("IOAPIC: RSDP virt=0x{:x}", rsdp_virt);

    let madt_info = match unsafe { x86_64::acpi::parse_madt(rsdp_virt, hhdm) } {
        Some(info) => info,
        None => {
            kernel::kdebug!("IOAPIC: Failed to parse MADT");
            return;
        }
    };
    kernel::kdebug!("IOAPIC: MADT parsed OK");

    // Store CPU IDs
    {
        let mut ids = [CpuId(0); x86_64::acpi::MAX_CPUS];
        for i in 0..madt_info.cpu_count {
            ids[i] = CpuId(madt_info.local_apic_ids[i]);
        }
        unsafe {
            x86_64::CPU_IDS = ids;
        }
        x86_64::CPU_COUNT.store(madt_info.cpu_count as u64, Ordering::SeqCst);
        kernel::kdebug!(
            "SMP: Found {} CPUs (CPU_COUNT now = {})",
            madt_info.cpu_count,
            x86_64::CPU_COUNT.load(Ordering::SeqCst)
        );
    }

    if madt_info.ioapic_count == 0 {
        kernel::kdebug!("IOAPIC: No IOAPICs found");
        return;
    }

    let ioapic = &madt_info.ioapics[0];
    kernel::kdebug!(
        "IOAPIC: Found at phys 0x{:08x}, GSI base {}",
        ioapic.mmio_base,
        ioapic.gsi_base
    );

    // 3. Initialize IOAPIC with HHDM offset
    x86_64::ioapic::init(ioapic.mmio_base, madt_info.local_apic_addr, hhdm);
    kernel::kdebug!("IOAPIC: Registers initialized");

    // Enable Local APIC (SVR, TPR)
    x86_64::ioapic::enable_local_apic();
    kernel::kdebug!("IOAPIC: Local APIC enabled (SVR=0x1FF, TPR=0)");

    let (version, max_entries) = x86_64::ioapic::get_version();
    kernel::kdebug!(
        "IOAPIC: version 0x{:02x}, {} redir entries",
        version,
        max_entries
    );

    x86_64::ioapic::mask_all();
    kernel::kdebug!("IOAPIC: All pins masked");

    // 4. Route IRQ1 (keyboard) -> vector 0x21
    let (gsi1, active_low1, level1) = madt_info.irq_to_gsi(1);
    let mut entry1 = x86_64::ioapic::RedirEntry::new_fixed(0x21, 0);
    entry1.active_low = active_low1;
    entry1.level_triggered = level1;
    x86_64::ioapic::write_redir(gsi1 as u8, entry1);
    x86_64::ioapic::unmask_pin(gsi1 as u8);
    kernel::ktrace!("IOAPIC: IRQ1 -> GSI {} -> 0x21", gsi1);

    // IRQ12 (mouse) -> vector 0x2C
    let (gsi12, active_low12, level12) = madt_info.irq_to_gsi(12);
    let mut entry12 = x86_64::ioapic::RedirEntry::new_fixed(0x2C, 0);
    entry12.active_low = active_low12;
    entry12.level_triggered = level12;
    x86_64::ioapic::write_redir(gsi12 as u8, entry12);
    x86_64::ioapic::unmask_pin(gsi12 as u8);
    kernel::ktrace!("IOAPIC: IRQ12 -> GSI {} -> 0x2C", gsi12);

    // IRQ4 (serial COM1) -> vector 0x24
    let (gsi4, active_low4, level4) = madt_info.irq_to_gsi(4);
    let mut entry4 = x86_64::ioapic::RedirEntry::new_fixed(0x24, 0);
    entry4.active_low = active_low4;
    entry4.level_triggered = level4;
    x86_64::ioapic::write_redir(gsi4 as u8, entry4);
    x86_64::ioapic::unmask_pin(gsi4 as u8);
    kernel::ktrace!("IOAPIC: IRQ4 -> GSI {} -> 0x24", gsi4);

    kernel::kdebug!("IOAPIC: Init complete");
}
