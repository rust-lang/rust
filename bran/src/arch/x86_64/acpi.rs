//! ACPI MADT Parsing
//!
//! Parse the Multiple APIC Description Table to discover:
//! - Local APICs (per-CPU)
//! - IOAPICs (interrupt controllers)
//! - Interrupt Source Overrides (legacy IRQ remapping)

use core::ptr;
use kernel::{FrameAllocatorHook, MapKind, MapPerms};

use super::paging;

struct AcpiMapAllocator;
impl FrameAllocatorHook for AcpiMapAllocator {
    fn alloc_frame(&self) -> Option<u64> {
        kernel::memory::alloc_frame()
    }
}

fn map_phys_range(phys: u64, len: u64, hhdm: u64) {
    let start = phys & !0xfff;
    let end = (phys + len + 0xfff) & !0xfff;
    let aspace = paging::active_address_space();
    let perms = MapPerms {
        user: false,
        read: true,
        write: false,
        exec: false,
        kind: MapKind::Normal,
    };

    let mut p = start;
    while p < end {
        let virt = p + hhdm;
        if paging::try_translate(aspace, virt).is_none() {
            let _ = paging::map_page(aspace, virt, p, perms, MapKind::Normal, &AcpiMapAllocator);
            paging::tlb_flush_page(virt);
        }
        p += 4096;
    }
}

fn rsdp_phys_from_virt(rsdp_addr: u64, hhdm_offset: u64) -> u64 {
    if rsdp_addr >= hhdm_offset {
        rsdp_addr - hhdm_offset
    } else {
        rsdp_addr
    }
}

/// ACPI table signature
const MADT_SIGNATURE: [u8; 4] = *b"APIC";

/// MADT entry types
const ENTRY_LOCAL_APIC: u8 = 0;
const ENTRY_IOAPIC: u8 = 1;
const ENTRY_INTERRUPT_OVERRIDE: u8 = 2;
#[allow(dead_code)]
const ENTRY_LOCAL_APIC_NMI: u8 = 4;
const ENTRY_LOCAL_X2APIC: u8 = 9;

/// Maximum supported CPUs
pub const MAX_CPUS: usize = 32;

/// Maximum supported IOAPICs
pub const MAX_IOAPICS: usize = 8;

/// Maximum interrupt source overrides
pub const MAX_ISO: usize = 24;

/// IOAPIC discovery info
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct IoapicInfo {
    pub id: u8,
    pub mmio_base: u64,
    pub gsi_base: u32,
}

/// Interrupt Source Override
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct InterruptOverride {
    pub bus: u8,
    pub source_irq: u8,
    pub gsi: u32,
    pub flags: u16, // bits 0-1: polarity, bits 2-3: trigger mode
}

impl InterruptOverride {
    /// Returns true if active low polarity
    pub fn is_active_low(&self) -> bool {
        let pol = self.flags & 0x03;
        pol == 0x03 // 11 = active low
    }

    /// Returns true if level triggered
    pub fn is_level_triggered(&self) -> bool {
        let trigger = (self.flags >> 2) & 0x03;
        trigger == 0x03 // 11 = level
    }
}

/// Result of MADT parsing
#[derive(Debug)]
#[repr(C, packed)]
struct LocalX2ApicEntry {
    entry_type: u8,
    length: u8,
    _reserved: u16,
    x2apic_id: u32,
    flags: u32,
    acpi_processor_id: u32,
}

pub struct MadtInfo {
    pub local_apic_addr: u64,
    pub ioapics: [IoapicInfo; MAX_IOAPICS],
    pub ioapic_count: usize,
    pub overrides: [InterruptOverride; MAX_ISO],
    pub override_count: usize,
    pub local_apic_ids: [u32; MAX_CPUS],
    pub cpu_count: usize,
}

impl MadtInfo {
    /// Look up GSI for a legacy IRQ, applying any overrides
    pub fn irq_to_gsi(&self, irq: u8) -> (u32, bool, bool) {
        // Check overrides first
        for i in 0..self.override_count {
            let iso = &self.overrides[i];
            if iso.source_irq == irq {
                return (iso.gsi, iso.is_active_low(), iso.is_level_triggered());
            }
        }
        // Default: identity mapping, edge triggered, active high
        (irq as u32, false, false)
    }

    /// Find which IOAPIC handles a given GSI
    #[allow(dead_code)]
    pub fn gsi_to_ioapic(&self, gsi: u32) -> Option<(usize, u8)> {
        for i in 0..self.ioapic_count {
            let ioapic = &self.ioapics[i];
            // Assume each IOAPIC handles 24 entries (typical)
            if gsi >= ioapic.gsi_base && gsi < ioapic.gsi_base + 24 {
                let pin = (gsi - ioapic.gsi_base) as u8;
                return Some((i, pin));
            }
        }
        None
    }
}

/// Parse the ACPI MADT from RSDP
///
/// # Safety
/// Caller must ensure hhdm_offset is valid and ACPI tables are mapped.
pub unsafe fn parse_madt(rsdp_virt: u64, hhdm_offset: u64) -> Option<MadtInfo> {
    let rsdp_phys = rsdp_phys_from_virt(rsdp_virt, hhdm_offset);
    map_phys_range(rsdp_phys, 4096, hhdm_offset);
    let rsdp = (rsdp_phys + hhdm_offset) as *const Rsdp;

    // Validate RSDP signature
    let sig = unsafe { ptr::read_unaligned(ptr::addr_of!((*rsdp).signature)) };
    if &sig != b"RSD PTR " {
        return None;
    }

    let revision = unsafe { ptr::read_unaligned(ptr::addr_of!((*rsdp).revision)) };

    let madt_phys = if revision >= 2 {
        // ACPI 2.0+: use XSDT
        let xsdt_phys = unsafe { ptr::read_unaligned(ptr::addr_of!((*rsdp).xsdt_address)) };
        map_phys_range(
            xsdt_phys,
            core::mem::size_of::<AcpiSdtHeader>() as u64,
            hhdm_offset,
        );
        let xsdt_virt = xsdt_phys + hhdm_offset;
        let length = unsafe {
            ptr::read_unaligned(ptr::addr_of!((*(xsdt_virt as *const AcpiSdtHeader)).length))
        };
        map_phys_range(xsdt_phys, length as u64, hhdm_offset);
        unsafe { find_table_xsdt(xsdt_virt, &MADT_SIGNATURE, hhdm_offset) }?
    } else {
        // ACPI 1.0: use RSDT
        let rsdt_phys = unsafe { ptr::read_unaligned(ptr::addr_of!((*rsdp).rsdt_address)) };
        map_phys_range(
            rsdt_phys as u64,
            core::mem::size_of::<AcpiSdtHeader>() as u64,
            hhdm_offset,
        );
        let rsdt_virt = (rsdt_phys as u64) + hhdm_offset;
        let length = unsafe {
            ptr::read_unaligned(ptr::addr_of!((*(rsdt_virt as *const AcpiSdtHeader)).length))
        };
        map_phys_range(rsdt_phys as u64, length as u64, hhdm_offset);
        unsafe { find_table_rsdt(rsdt_virt, &MADT_SIGNATURE, hhdm_offset) }?
    };

    let madt_virt = madt_phys + hhdm_offset;
    map_phys_range(
        madt_phys,
        core::mem::size_of::<AcpiSdtHeader>() as u64,
        hhdm_offset,
    );
    let madt_length = unsafe {
        ptr::read_unaligned(ptr::addr_of!((*(madt_virt as *const AcpiSdtHeader)).length))
    };
    kernel::kdebug!("MADT: total length {}", madt_length);
    map_phys_range(madt_phys, madt_length as u64, hhdm_offset);
    unsafe { parse_madt_table(madt_virt) }
}

unsafe fn find_table_rsdt(rsdt_virt: u64, sig: &[u8; 4], hhdm: u64) -> Option<u64> {
    let header = rsdt_virt as *const AcpiSdtHeader;
    let length = unsafe { ptr::read_unaligned(ptr::addr_of!((*header).length)) };
    let entry_count = (length as usize - 36) / 4;
    let entries = (rsdt_virt + 36) as *const u32;

    for i in 0..entry_count {
        let addr = unsafe { ptr::read_unaligned(entries.add(i)) } as u64;
        map_phys_range(addr, core::mem::size_of::<AcpiSdtHeader>() as u64, hhdm);
        let table_header = (addr + hhdm) as *const AcpiSdtHeader;
        let table_sig = unsafe { ptr::read_unaligned(ptr::addr_of!((*table_header).signature)) };
        if &table_sig == sig {
            return Some(addr);
        }
    }
    None
}

unsafe fn find_table_xsdt(xsdt_virt: u64, sig: &[u8; 4], hhdm: u64) -> Option<u64> {
    let header = xsdt_virt as *const AcpiSdtHeader;
    let length = unsafe { ptr::read_unaligned(ptr::addr_of!((*header).length)) };
    let entry_count = (length as usize - 36) / 8;
    let entries = (xsdt_virt + 36) as *const u64;

    for i in 0..entry_count {
        let addr = unsafe { ptr::read_unaligned(entries.add(i)) };
        map_phys_range(addr, core::mem::size_of::<AcpiSdtHeader>() as u64, hhdm);
        let table_header = (addr + hhdm) as *const AcpiSdtHeader;
        let table_sig = unsafe { ptr::read_unaligned(ptr::addr_of!((*table_header).signature)) };
        if &table_sig == sig {
            return Some(addr);
        }
    }
    None
}

unsafe fn parse_madt_table(madt_virt: u64) -> Option<MadtInfo> {
    let header = madt_virt as *const MadtHeader;
    let length = unsafe { ptr::read_unaligned(ptr::addr_of!((*header).header.length)) };
    let local_apic_addr = unsafe { ptr::read_unaligned(ptr::addr_of!((*header).local_apic_addr)) };

    let mut info = MadtInfo {
        local_apic_addr: local_apic_addr as u64,
        ioapics: [IoapicInfo::default(); MAX_IOAPICS],
        ioapic_count: 0,
        overrides: [InterruptOverride::default(); MAX_ISO],
        override_count: 0,
        local_apic_ids: [0; MAX_CPUS],
        cpu_count: 0,
    };

    let entries_start = madt_virt + 44; // sizeof(MadtHeader)
    let entries_end = madt_virt + length as u64;
    let mut ptr = entries_start;

    while ptr + 2 <= entries_end {
        let entry_type = unsafe { ptr::read_unaligned(ptr as *const u8) };
        let entry_len = unsafe { ptr::read_unaligned((ptr + 1) as *const u8) };

        // Debug: log EVERY entry
        kernel::ktrace!(
            "MADT: Entry type {}, len {} at 0x{:x}",
            entry_type,
            entry_len,
            ptr
        );

        if entry_len < 2 {
            kernel::kerror!("MADT: Invalid entry length {} at 0x{:x}", entry_len, ptr);
            break;
        }

        match entry_type {
            ENTRY_LOCAL_APIC => {
                let entry = ptr as *const LocalApicEntry;
                let flags = unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).flags)) };
                if (flags & 1) != 0 || (flags & 2) != 0 {
                    let apic_id = unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).apic_id)) };
                    if info.cpu_count < MAX_CPUS {
                        info.local_apic_ids[info.cpu_count] = apic_id as u32;
                        info.cpu_count += 1;
                    } else {
                        kernel::kwarn!("MADT: Too many CPUs, ignoring APIC ID {}", apic_id);
                    }
                }
            }
            ENTRY_LOCAL_X2APIC => {
                let entry = ptr as *const LocalX2ApicEntry;
                let flags = unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).flags)) };
                if (flags & 1) != 0 || (flags & 2) != 0 {
                    let apic_id = unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).x2apic_id)) };
                    if info.cpu_count < MAX_CPUS {
                        info.local_apic_ids[info.cpu_count] = apic_id;
                        info.cpu_count += 1;
                    } else {
                        kernel::kwarn!("MADT: Too many CPUs, ignoring x2APIC ID {}", apic_id);
                    }
                }
            }
            ENTRY_IOAPIC if info.ioapic_count < MAX_IOAPICS => {
                let entry = ptr as *const IoapicEntry;
                info.ioapics[info.ioapic_count] = IoapicInfo {
                    id: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).ioapic_id)) },
                    mmio_base: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).ioapic_addr)) }
                        as u64,
                    gsi_base: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).gsi_base)) },
                };
                info.ioapic_count += 1;
            }
            ENTRY_INTERRUPT_OVERRIDE if info.override_count < MAX_ISO => {
                let entry = ptr as *const InterruptOverrideEntry;
                info.overrides[info.override_count] = InterruptOverride {
                    bus: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).bus)) },
                    source_irq: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).source)) },
                    gsi: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).gsi)) },
                    flags: unsafe { ptr::read_unaligned(ptr::addr_of!((*entry).flags)) },
                };
                info.override_count += 1;
            }
            _ => {}
        }

        ptr += entry_len as u64;
    }

    Some(info)
}

// ACPI structures

#[repr(C, packed)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
    // ACPI 2.0+ fields
    length: u32,
    xsdt_address: u64,
    ext_checksum: u8,
    reserved: [u8; 3],
}

#[repr(C, packed)]
struct AcpiSdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

#[repr(C, packed)]
struct MadtHeader {
    header: AcpiSdtHeader,
    local_apic_addr: u32,
    flags: u32,
}

#[repr(C, packed)]
struct LocalApicEntry {
    entry_type: u8,
    length: u8,
    processor_id: u8,
    apic_id: u8,
    flags: u32,
}

#[repr(C, packed)]
struct IoapicEntry {
    entry_type: u8,
    length: u8,
    ioapic_id: u8,
    reserved: u8,
    ioapic_addr: u32,
    gsi_base: u32,
}

#[repr(C, packed)]
struct InterruptOverrideEntry {
    entry_type: u8,
    length: u8,
    bus: u8,
    source: u8,
    gsi: u32,
    flags: u16,
}
