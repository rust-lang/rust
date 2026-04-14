use crate::runtime::ArchRuntime;
use core::arch::asm;
use kernel::time::MonotonicClamp;
use kernel::{FrameAllocatorHook, IrqState, MapKind, MapPerms, UserEntry, UserTaskSpec};

pub mod paging;
pub mod serial;
pub mod task;
pub mod trap;
pub mod vector;

pub struct RISCV64Runtime {
    serial: serial::SerialPort,
    clamp: MonotonicClamp,
}

impl RISCV64Runtime {
    pub const fn new() -> Self {
        Self {
            serial: serial::SerialPort::new(),
            clamp: MonotonicClamp::new(),
        }
    }
}

pub use paging::RISCV64AddressSpace;
pub use task::RISCV64Context;

impl ArchRuntime for RISCV64Runtime {
    type Context = RISCV64Context;
    type AddressSpace = RISCV64AddressSpace;

    fn init(&self, hhdm_offset: u64) {
        paging::init(hhdm_offset);
        unsafe {
            vector::init();
        }
    }
    fn putchar(&self, c: u8) {
        self.serial.putchar(c);
    }
    fn getchar(&self) -> Option<u8> {
        self.serial.getchar()
    }
    fn halt(&self) -> ! {
        hcf()
    }

    fn reboot(&self) -> ! {
        // SBI System Reset Extension (SRST)
        // EID = 0x53525354 ("SRST"), FID = 0 (sbi_system_reset)
        // reset_type = 0 (shutdown), but we want 1 (cold reboot)
        unsafe {
            asm!(
                "ecall",
                in("a7") 0x53525354u64,  // EID: SRST
                in("a6") 0u64,           // FID: sbi_system_reset
                in("a0") 1u64,           // reset_type: cold reboot
                in("a1") 0u64,           // reset_reason: no reason
                options(noreturn)
            );
        }
    }

    fn wait_for_interrupt(&self) {
        unsafe {
            core::arch::asm!("wfi");
        }
    }

    fn mono_ticks(&self) -> u64 {
        let raw = read_time();
        self.clamp.clamp(raw)
    }

    fn mono_freq_hz(&self) -> u64 {
        10_000_000
    }

    fn irq_disable(&self) -> IrqState {
        let sstatus: usize;
        unsafe {
            asm!("csrrci {}, sstatus, 2", out(reg) sstatus);
        }
        IrqState((sstatus >> 1) & 1)
    }

    fn irq_restore(&self, state: IrqState) {
        if state.0 != 0 {
            unsafe {
                asm!("csrrs x0, sstatus, 2");
            }
        } else {
            unsafe {
                asm!("csrrci x0, sstatus, 2");
            }
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
        // RISC-V user mode entry via sret
        // sepc = entry.entry_pc
        // sstatus: Clear SPP (bit 8) -> User Mode
        //          Set SPIE (bit 5) -> Interrupts enabled after sret (if we want them)
        //          Wait, user prompt: "if unstable, start with interrupts masked".
        //          So allow timer IRQs... "after entry only when trap path known-good"
        //          So maybe zero SPIE for now?
        //          "set SPIE appropriately"
        //          Let's Set SPIE to 1 (enabled) usually, but maybe 0 for safety first?
        //          We'll set SPIE=1 so we don't block interrupts forever if we can handle them.
        //          Actually, "interrupts masked in user until..." => SPIE=0.
        //          If SPIE=0, after sret, SIE (bit 1) becomes 0.
        //          Safe choice: SPIE=0.

        let mut sstatus: usize;
        unsafe {
            asm!("csrr {}, sstatus", out(reg) sstatus);
        }
        sstatus &= !(1 << 8); // Clear SPP (User)
        sstatus &= !(1 << 5); // Clear SPIE (Disable interrupts in user mode for now)
        // Note: bit 1 (SIE) is preserved for Supervisor, but overwritten by SPIE into SIE on sret.

        unsafe {
            asm!(
                "csrw sstatus, {sstatus}",
                "csrw sepc, {pc}",
                "mv sp, {sp}",
                "mv a0, {arg}",
                "sret",
                sstatus = in(reg) sstatus,
                pc = in(reg) entry.entry_pc,
                sp = in(reg) entry.user_sp,
                arg = in(reg) entry.arg0,
                options(noreturn)
            );
        }
    }

    // Paging
    fn make_user_address_space(&self) -> Self::AddressSpace {
        paging::make_user_address_space(self.active_address_space(), &DumbKernelAlloc)
    }

    fn active_address_space(&self) -> Self::AddressSpace {
        paging::active_address_space()
    }

    fn activate_address_space(&self, aspace: Self::AddressSpace) {
        let satp = (8 << 60) | (aspace.0 >> 12); // Sv39
        unsafe {
            asm!("csrw satp, {}", in(reg) satp);
            asm!("sfence.vma");
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
        unsafe {
            asm!("wfi");
        }
    }
}

#[inline]
fn read_time() -> u64 {
    let val: u64;
    unsafe {
        asm!("csrr {}, time", out(reg) val);
    }
    val
}
