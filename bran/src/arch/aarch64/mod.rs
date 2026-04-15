use crate::runtime::ArchRuntime;
use core::arch::{asm, naked_asm};
use kernel::time::MonotonicClamp;
use kernel::{FrameAllocatorHook, IrqState, MapKind, MapPerms, UserEntry, UserTaskSpec};

pub mod paging;
pub mod simd;
pub mod syscall;
pub mod task;
pub mod trap;
pub mod vector;

pub struct AArch64Runtime {
    serial: SerialPort,
}

impl AArch64Runtime {
    pub const fn new() -> Self {
        Self {
            serial: SerialPort::new(),
        }
    }
}

pub use paging::AArch64AddressSpace;
pub use task::AArch64Context;

/// Ensure execution is in EL1h (using SP_EL1) before normal kernel init.
///
/// Handles two boot scenarios:
///
/// 1. **Arriving at EL2** (some firmware/hypervisor configurations): sets
///    `HCR_EL2.RW=1` so that EL1 runs as AArch64, programs `SPSR_EL2` for
///    EL1h with D/A/I/F masked, sets `ELR_EL2` to the post-`eret`
///    continuation, and executes `eret` to drop to EL1h.
///
/// 2. **Already at EL1t** (SPSel=0): copies the current SP value into
///    `SP_EL1` and sets `SPSel=1`.
///
/// If `SPSel` is already 1 the function is a no-op (idempotent).
#[unsafe(naked)]
unsafe extern "C" fn switch_to_el1h() {
    naked_asm!(
        // Save current SP before any branching so both paths can initialise
        // SP_EL1 to a valid kernel stack address.
        "mov  x9, sp",

        // ── Detect current EL ───────────────────────────────────────────
        "mrs  x10, CurrentEL",
        "lsr  x10, x10, #2",   // bits [3:2] → x10
        "and  x10, x10, #3",   // isolate two-bit EL field
        "cmp  x10, #2",
        "bne  1f",             // not EL2 → already at EL1, go to common path

        // ── EL2 path ────────────────────────────────────────────────────
        // HCR_EL2.RW = 1  →  EL1/EL0 execute as AArch64.
        "mrs  x10, hcr_el2",
        "orr  x10, x10, #(1 << 31)",
        "msr  hcr_el2, x10",

        // SPSR_EL2:
        //   M[4:0] = 0b00101 = 5  (EL1h, using SP_EL1)
        //   D/A/I/F (bits [9:6]) = 0b1111  (all async aborts + IRQ + FIQ masked)
        //   → 0x3C5
        "mov  x10, #0x3C5",
        "msr  spsr_el2, x10",

        // ELR_EL2: resume at the common SP_EL1 setup block after eret.
        "adr  x10, 2f",
        "msr  elr_el2, x10",

        "isb",
        "eret",                // → EL1h, continues at label 2

        // ── EL1 path (fall-through) ──────────────────────────────────────
        "1:",
        // ── Common: install saved SP into SP_EL1 and select it ──────────
        "2:",
        "msr  spsel, #1",      // switch stack-pointer select to SP_EL1
        "mov  sp, x9",         // initialise SP_EL1 from the saved value
        "isb",
        "ret",
    );
}

impl ArchRuntime for AArch64Runtime {
    type Context = AArch64Context;
    type AddressSpace = AArch64AddressSpace;

    fn init(&self, hhdm_offset: u64) {
        paging::init(hhdm_offset);
        unsafe {
            vector::init();
        }
    }
    fn putchar(&self, c: u8) {
        self.serial.putchar(c);
    }
    // getchar: default None (semihosting has no standard getchar)

    fn halt(&self) -> ! {
        hcf()
    }

    fn reboot(&self) -> ! {
        // PSCI SYSTEM_RESET via HVC
        unsafe {
            asm!(
                "mov x0, {fid}",
                "hvc #0",
                fid = in(reg) 0x8400_0009u64,
                options(noreturn)
            );
        }
    }

    fn wait_for_interrupt(&self) {
        unsafe {
            core::arch::asm!("wfe", options(nomem, nostack));
        }
    }

    fn mono_ticks(&self) -> u64 {
        self.serial.clamp.clamp(read_cntvct_el0())
    }

    fn mono_freq_hz(&self) -> u64 {
        read_cntfrq_el0()
    }

    fn irq_disable(&self) -> IrqState {
        let daif: u64;
        unsafe {
            asm!("mrs {}, daif", out(reg) daif, options(nomem, nostack));
            asm!("msr daifset, #2", options(nomem, nostack));
        }
        IrqState((daif >> 7) as usize & 1)
    }

    fn irq_restore(&self, state: IrqState) {
        if state.0 == 0 {
            unsafe {
                asm!("msr daifclr, #2", options(nomem, nostack));
            }
        } else {
            unsafe {
                asm!("msr daifset, #2", options(nomem, nostack));
            }
        }
    }

    fn simd_init_cpu(&self) {
        simd::init_cpu();
    }
    fn simd_state_layout(&self) -> (usize, usize) {
        simd::STATE_LAYOUT
    }
    unsafe fn simd_save(&self, dst: *mut u8) {
        unsafe { simd::save(dst) }
    }
    unsafe fn simd_restore(&self, src: *const u8) {
        unsafe { simd::restore(src) }
    }

    unsafe fn early_init(&self) {
        unsafe {
            switch_to_el1h();
        }
    }

    fn fence_full(&self) {
        unsafe {
            asm!("dmb sy", options(nostack, preserves_flags));
        }
    }

    fn icache_invalidate(&self) {
        unsafe {
            asm!("ic ialluis", options(nostack, preserves_flags));
            asm!("dsb ish", options(nostack, preserves_flags));
            asm!("isb", options(nostack, preserves_flags));
        }
    }

    fn threads_supported(&self) -> bool {
        true
    }

    // Tasking
    fn init_kernel_context(
        &self,
        entry: extern "C" fn(usize) -> !,
        stack_top: u64,
        arg: usize,
    ) -> Self::Context {
        task::init_kernel_context(entry, stack_top, arg)
    }

    fn init_user_context(
        &self,
        spec: UserTaskSpec<Self::AddressSpace>,
        kstack_top: u64,
    ) -> Self::Context {
        task::init_user_context(spec, kstack_top)
    }

    unsafe fn switch(&self, from: &mut Self::Context, to: &Self::Context, _to_tid: u64) {
        unsafe { task::switch(from, to) }
    }

    unsafe fn enter_user(&self, entry: UserEntry) -> ! {
        // Debug: Read current TTBR0
        let ttbr0: u64;
        unsafe {
            asm!("mrs {}, ttbr0_el1", out(reg) ttbr0, options(nomem, nostack));
        }
        kernel::kinfo!(
            "enter_user: TTBR0={:#x} entry_pc={:#x} user_sp={:#x}",
            ttbr0,
            entry.entry_pc,
            entry.user_sp
        );

        // Switch to EL1h (using SP_EL1) so we can safely set SP_EL0 for user mode.
        // We first save the current SP, then switch SPSel=1 and restore SP to SP_EL1.
        // After this, SP_EL0 can be safely written for the user task.
        //
        // SPSR: EL0t (mode 0), all interrupts unmasked
        let spsr: u64 = 0;
        let ksp: u64;
        unsafe {
            asm!("mov {}, sp", out(reg) ksp, options(nomem, nostack));
        }

        unsafe {
            asm!(
                "msr spsel, #1",
                "mov sp, {ksp}",
                "msr sp_el0, {sp}",
                "msr elr_el1, {pc}",
                "msr spsr_el1, {spsr}",
                "mov x0, {arg}",
                "eret",
                ksp = in(reg) ksp,
                sp = in(reg) entry.user_sp,
                pc = in(reg) entry.entry_pc,
                spsr = in(reg) spsr,
                arg = in(reg) entry.arg0,
                options(noreturn)
            );
        }
    }

    // Paging - use ProxyAllocator for real page table allocation
    fn make_user_address_space(&self) -> Self::AddressSpace {
        let aspace = paging::make_user_address_space(self.active_address_space(), &ProxyAllocator);
        kernel::kinfo!(
            "make_user_address_space: created aspace phys={:#x}",
            aspace.0
        );
        aspace
    }

    fn active_address_space(&self) -> Self::AddressSpace {
        paging::active_address_space()
    }

    fn activate_address_space(&self, aspace: Self::AddressSpace) {
        kernel::kinfo!("activate_address_space: setting TTBR0 to {:#x}", aspace.0);
        unsafe {
            asm!(
                "msr ttbr0_el1, {ttbr}",
                "isb",
                "tlbi vmalle1is",
                "dsb ish",
                "isb",
                ttbr = in(reg) aspace.0,
                options(nostack)
            );
        }
    }

    fn map_page(
        &self,
        aspace: Self::AddressSpace,
        virt: u64,
        phys: u64,
        perms: MapPerms,
        kind: MapKind,
        allocator: &dyn FrameAllocatorHook,
    ) -> Result<(), ()> {
        paging::map_page(aspace, virt, phys, perms, kind, allocator)
    }

    fn unmap_page(&self, aspace: Self::AddressSpace, virt: u64) -> Result<Option<u64>, ()> {
        paging::unmap_page(aspace, virt)
    }

    fn translate(&self, aspace: Self::AddressSpace, virt: u64) -> Option<u64> {
        paging::translate(aspace, virt)
    }

    fn tlb_flush_page(&self, virt: u64) {
        paging::tlb_flush_page(virt)
    }

    fn aspace_to_raw(&self, aspace: Self::AddressSpace) -> u64 {
        aspace.0
    }
}
impl FrameAllocatorHook for ProxyAllocator {
    fn alloc_frame(&self) -> Option<u64> {
        kernel::memory::alloc_frame()
    }
}

pub struct SerialPort {
    pub clamp: MonotonicClamp,
}

impl SerialPort {
    pub const fn new() -> Self {
        Self {
            clamp: MonotonicClamp::new(),
        }
    }

    fn putchar(&self, c: u8) {
        // Semihosting SYS_WRITEC operation
        let ch = c;
        unsafe {
            asm!(
                "hlt #0xF000",
                in("w0") 0x03,
                in("x1") &ch,
                options(nostack, preserves_flags)
            );
        }
    }
}

pub fn hcf() -> ! {
    loop {
        unsafe {
            asm!("wfi", options(nomem, nostack));
        }
    }
}

#[inline]
fn read_cntfrq_el0() -> u64 {
    let val: u64;
    unsafe {
        asm!("mrs {}, cntfrq_el0", out(reg) val, options(nomem, nostack));
    }
    val
}

#[inline]
fn read_cntvct_el0() -> u64 {
    let val: u64;
    unsafe {
        asm!("mrs {}, cntvct_el0", out(reg) val, options(nomem, nostack));
    }
    val
}
