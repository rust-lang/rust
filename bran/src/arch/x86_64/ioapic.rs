//! IOAPIC Programming
//!
//! Memory-mapped access to I/O APIC for interrupt routing.

use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};
use kernel::{FrameAllocatorHook, MapKind, MapPerms, ioport_read_u8, ioport_write_u8};

use super::paging;

struct IoapicMapAllocator;
impl FrameAllocatorHook for IoapicMapAllocator {
    fn alloc_frame(&self) -> Option<u64> {
        kernel::memory::alloc_frame()
    }
}

fn map_mmio_range(phys: u64, len: u64, hhdm: u64) {
    let start = phys & !0xfff;
    let end = (phys + len + 0xfff) & !0xfff;
    let aspace = paging::active_address_space();
    let perms = MapPerms {
        user: false,
        read: true,
        write: true,
        exec: false,
        kind: MapKind::Device,
    };

    let mut p = start;
    while p < end {
        let virt = p + hhdm;
        if paging::try_translate(aspace, virt).is_none() {
            let _ = paging::map_page(aspace, virt, p, perms, MapKind::Device, &IoapicMapAllocator);
            paging::tlb_flush_page(virt);
        }
        p += 4096;
    }
}

/// IOAPIC register offsets
#[allow(dead_code)]
const IOREGSEL: u64 = 0x00;
const IOWIN: u64 = 0x10;

/// IOAPIC registers (via indirect access)
#[allow(dead_code)]
const IOAPIC_ID: u32 = 0x00;
const IOAPIC_VER: u32 = 0x01;
#[allow(dead_code)]
const IOAPIC_ARB: u32 = 0x02;
const IOAPIC_REDTBL_BASE: u32 = 0x10;
const LAPIC_LVT_TIMER: u32 = 0x320;
const LAPIC_TIMER_INITCNT: u32 = 0x380;
const LAPIC_TIMER_DIVIDE: u32 = 0x3E0;

/// Delivery modes for redirection entries
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
pub enum DeliveryMode {
    Fixed = 0,
    LowestPriority = 1,
    Smi = 2,
    Nmi = 4,
    Init = 5,
    ExtInt = 7,
}

/// Global IOAPIC state - only one IOAPIC supported for now
static IOAPIC_BASE: AtomicU64 = AtomicU64::new(0);
static HHDM_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Local APIC base for EOI
static LOCAL_APIC_BASE: AtomicU64 = AtomicU64::new(0xFEE00000);

/// Initialize IOAPIC with discovered MMIO base
pub fn init(mmio_base: u64, local_apic: u64, hhdm: u64) {
    IOAPIC_BASE.store(mmio_base, Ordering::SeqCst);
    HHDM_OFFSET.store(hhdm, Ordering::SeqCst);
    LOCAL_APIC_BASE.store(local_apic, Ordering::SeqCst);

    // Ensure MMIO ranges are mapped in the HHDM.
    map_mmio_range(mmio_base, 0x20, hhdm);
    map_mmio_range(local_apic, 0x1000, hhdm);
}

fn base() -> u64 {
    let phys = IOAPIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    phys + hhdm
}

/// Read IOAPIC register via indirect access
fn read_reg(reg: u32) -> u32 {
    unsafe {
        let base = base();
        ptr::write_volatile(base as *mut u32, reg);
        ptr::read_volatile((base + IOWIN) as *const u32)
    }
}

/// Write IOAPIC register via indirect access  
fn write_reg(reg: u32, val: u32) {
    unsafe {
        let base = base();
        ptr::write_volatile(base as *mut u32, reg);
        ptr::write_volatile((base + IOWIN) as *mut u32, val);
    }
}

/// Get IOAPIC version and max redirection entries
pub fn get_version() -> (u8, u8) {
    let ver = read_reg(IOAPIC_VER);
    let version = (ver & 0xFF) as u8;
    let max_redir = ((ver >> 16) & 0xFF) as u8;
    (version, max_redir + 1)
}

/// Redirection table entry (64-bit)
#[derive(Debug, Clone, Copy)]
pub struct RedirEntry {
    pub vector: u8,
    pub delivery_mode: DeliveryMode,
    pub dest_logical: bool,
    pub active_low: bool,
    pub level_triggered: bool,
    pub mask: bool,
    pub destination: u8,
}

impl RedirEntry {
    /// Create a new redirection entry for a fixed interrupt
    pub fn new_fixed(vector: u8, dest_cpu: u8) -> Self {
        Self {
            vector,
            delivery_mode: DeliveryMode::Fixed,
            dest_logical: false,
            active_low: false,
            level_triggered: false,
            mask: false,
            destination: dest_cpu,
        }
    }

    /// Convert to 64-bit register value
    fn to_u64(&self) -> u64 {
        let mut val: u64 = 0;
        val |= self.vector as u64;
        val |= (self.delivery_mode as u64) << 8;
        if self.dest_logical {
            val |= 1 << 11;
        }
        if self.active_low {
            val |= 1 << 13;
        }
        if self.level_triggered {
            val |= 1 << 15;
        }
        if self.mask {
            val |= 1 << 16;
        }
        val |= (self.destination as u64) << 56;
        val
    }
}

/// Write a redirection table entry
pub fn write_redir(pin: u8, entry: RedirEntry) {
    let reg_low = IOAPIC_REDTBL_BASE + (pin as u32 * 2);
    let reg_high = reg_low + 1;
    let val = entry.to_u64();

    write_reg(reg_high, (val >> 32) as u32);
    write_reg(reg_low, val as u32);
}

/// Mask an IOAPIC pin
pub fn mask_pin(pin: u8) {
    let reg_low = IOAPIC_REDTBL_BASE + (pin as u32 * 2);
    let mut val = read_reg(reg_low);
    val |= 1 << 16; // Set mask bit
    write_reg(reg_low, val);
}

/// Unmask an IOAPIC pin
pub fn unmask_pin(pin: u8) {
    let reg_low = IOAPIC_REDTBL_BASE + (pin as u32 * 2);
    let mut val = read_reg(reg_low);
    val &= !(1 << 16); // Clear mask bit
    write_reg(reg_low, val);
}

/// Send End-of-Interrupt to Local APIC
/// Must be called at the end of every interrupt handler
pub fn send_eoi() {
    let lapic_base = LOCAL_APIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    let eoi_reg = lapic_base + hhdm + 0xB0; // EOI register at offset 0xB0
    unsafe {
        ptr::write_volatile(eoi_reg as *mut u32, 0);
    }
}

/// Calibrate LAPIC timer against PIT to determine ticks per second.
pub fn calibrate_lapic_timer(hz: u32) -> (u32, u64) {
    let lapic_base = LOCAL_APIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    let base = lapic_base + hhdm;

    unsafe {
        // 1. Set divide configuration to 16
        ptr::write_volatile((base + LAPIC_TIMER_DIVIDE as u64) as *mut u32, 0x03);

        // 2. Calibrate LAPIC timer against PIT
        ioport_write_u8(0x43, 0xB0);
        let count = 11932u16;
        ioport_write_u8(0x42, (count & 0xFF) as u8);
        ioport_write_u8(0x42, (count >> 8) as u8);

        let port61 = ioport_read_u8(0x61);
        ioport_write_u8(0x61, port61 | 0x01);

        ptr::write_volatile((base + LAPIC_TIMER_INITCNT as u64) as *mut u32, 0xFFFFFFFF);
        let start_lapic = ptr::read_volatile((base + 0x390) as *const u32);

        let mut timeout = 10_000_000;
        while (ioport_read_u8(0x61) & 0x20) == 0 {
            core::hint::spin_loop();
            timeout -= 1;
            if timeout == 0 {
                break;
            }
        }

        let end_lapic = ptr::read_volatile((base + 0x390) as *const u32);
        ioport_write_u8(0x61, port61 & !0x01);

        let delta = start_lapic.saturating_sub(end_lapic);
        let ticks_per_sec = (delta as u64) * 100;
        let init_cnt = (ticks_per_sec / hz as u64) as u32;

        (init_cnt, ticks_per_sec)
    }
}

/// Set LAPIC timer to periodic mode with a pre-calculated initial count.
/// This avoids re-calibrating against the PIT on every CPU.
pub fn set_lapic_timer_periodic(vector: u8, init_cnt: u32) {
    let lapic_base = LOCAL_APIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    let base = lapic_base + hhdm;

    unsafe {
        // 1. Set divide configuration to 16
        ptr::write_volatile((base + LAPIC_TIMER_DIVIDE as u64) as *mut u32, 0x03);

        // 2. Set LVT Timer register: Periodic mode (bit 17) + Vector
        let lvt_val = (1 << 17) | (vector as u32);
        ptr::write_volatile((base + LAPIC_LVT_TIMER as u64) as *mut u32, lvt_val);

        // 3. Set final initial count
        ptr::write_volatile((base + LAPIC_TIMER_INITCNT as u64) as *mut u32, init_cnt);
    }
}

/// Send a Fixed IPI to a target CPU's Local APIC.
pub fn send_fixed_ipi(apic_id: u32, vector: u8) {
    let lapic_base = LOCAL_APIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    let base = lapic_base + hhdm;

    unsafe {
        // ICR High: Destination (bits 56-63)
        ptr::write_volatile((base + 0x310) as *mut u32, apic_id << 24);
        // ICR Low: Fixed (mode 0), Edge (0), Physical (0), Vector (bits 0-7)
        // Bit 14 is 'Assert' (usually 1 for fixed IPIs)
        ptr::write_volatile((base + 0x300) as *mut u32, (1 << 14) | (vector as u32));
    }
}

pub fn lapic_in_service_vector() -> Option<u8> {
    let lapic_base = LOCAL_APIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    let base = lapic_base + hhdm;

    for i in (0..8).rev() {
        let reg = base + 0x100 + (i * 0x10) as u64;
        let val = unsafe { ptr::read_volatile(reg as *const u32) };
        if val != 0 {
            let bit = 31 - val.leading_zeros();
            let vec = (i * 32 + bit) as u8;
            return Some(vec);
        }
    }
    None
}

/// Mask all IOAPIC pins (for initialization)
pub fn mask_all() {
    let (_, max_entries) = get_version();
    for pin in 0..max_entries {
        mask_pin(pin);
    }
}

/// Explicitly enable the Local APIC
/// This is required because relying on BIOS state is unreliable.
/// Sets SVR (0xF0) to Enable + Vector 0xFF.
/// Sets TPR (0x80) to 0 (Accept all).
pub fn enable_local_apic() {
    let lapic_base = LOCAL_APIC_BASE.load(Ordering::SeqCst);
    let hhdm = HHDM_OFFSET.load(Ordering::SeqCst);
    let base = lapic_base + hhdm;

    unsafe {
        // 1. Spurious Interrupt Vector Register (0xF0)
        // Bit 8: Enable APIC
        // Bits 0-7: Vector (0xFF is common for spurious)
        let svr = 0x100 | 0xFF;
        ptr::write_volatile((base + 0xF0) as *mut u32, svr);

        // 2. Task Priority Register (0x80)
        // Set to 0 to accept all priorities
        ptr::write_volatile((base + 0x80) as *mut u32, 0);

        // 3. Logical Destination Register (0xD0)
        // Set ID to 1 (Logical ID for this CPU in Flat Mode)
        // This assumes Flat Model. For Physical mode routing (which we use), this is less critical
        // but good for sanity.
        // val = (Logical ID << 24)
        // We'll skip this for now since we use Physical Destination Mode in IOAPIC.

        // 4. Destination Format Register (0xE0)
        // Set to Flat Model (0xFFFFFFFF)
        ptr::write_volatile((base + 0xE0) as *mut u32, 0xFFFFFFFF);

        // 5. Acknowledge any pending EOI just in case
        ptr::write_volatile((base + 0xB0) as *mut u32, 0);
    }
}
