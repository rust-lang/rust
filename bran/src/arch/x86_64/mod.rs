// Updated X86_64 runtime with user entry mapping and alignment check
use crate::runtime::ArchRuntime;
use core::arch::asm;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use kernel::once_cell::OnceCell;
use kernel::time::MonotonicClamp;
use kernel::{CpuId, FrameAllocatorHook, IrqState, MapKind, MapPerms, UserEntry, UserTaskSpec};

pub mod acpi;
pub mod apic;
pub mod cmos;
pub mod entropy;
pub mod gdt;
pub mod idt;
pub mod ioapic;
pub mod paging;
pub mod pci;
pub mod pic;
pub mod simd;
pub mod smp;
pub mod syscall;
pub mod task;
pub mod tls;
pub mod trap;

pub struct X86_64Runtime {
    clamp: MonotonicClamp,
    pci_root: AtomicU64,
    hhdm_offset: AtomicU64,
    cpu_ids: OnceCell<&'static [CpuId]>,

    // Timer calibration results for SMP sync
    timer_vector: AtomicUsize,
    timer_init_cnt: AtomicUsize,

    started_cpu_count: AtomicUsize,
    trampoline_ready: AtomicUsize,

    serial_buf: spin::Mutex<SerialBuffer>,
}

pub struct SerialBuffer {
    data: [u8; 1024],
    head: usize,
    tail: usize,
}

impl SerialBuffer {
    pub const fn new() -> Self {
        Self {
            data: [0; 1024],
            head: 0,
            tail: 0,
        }
    }

    pub fn push(&mut self, val: u8) {
        let next_head = (self.head + 1) % self.data.len();
        if next_head != self.tail {
            self.data[self.head] = val;
            self.head = next_head;
        }
    }

    pub fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail {
            None
        } else {
            let val = self.data[self.tail];
            self.tail = (self.tail + 1) % self.data.len();
            Some(val)
        }
    }
}

pub static mut CPU_IDS: [CpuId; acpi::MAX_CPUS] = [CpuId(0); acpi::MAX_CPUS];
pub static CPU_COUNT: AtomicU64 = AtomicU64::new(1); // Default to 1 (BSP)

impl X86_64Runtime {
    const LAPIC_ICR_DELIVERY_TIMEOUT_US: u64 = 100_000;
    const AP_STARTUP_TIMEOUT_US: u64 = 5_000_000;
    const AP_FLAG_POLL_INTERVAL_US: u64 = 25;

    pub fn cpu_ids(&self) -> &'static [CpuId] {
        if !self.cpu_ids.is_initialized() {
            let count = CPU_COUNT.load(Ordering::SeqCst) as usize;
            self.cpu_ids.set(unsafe { &CPU_IDS[..count] });
        }
        *self.cpu_ids.get()
    }

    pub fn current_cpu_id(&self) -> CpuId {
        match self.lapic_id() {
            Ok(id) => CpuId(id),
            Err(_) => CpuId(0),
        }
    }

    pub fn poll_serial(&self) {
        let mut buf = self.serial_buf.lock();
        self.poll_serial_unlocked(&mut buf);
    }

    fn poll_serial_unlocked(&self, buf: &mut SerialBuffer) {
        loop {
            let lsr: u8;
            unsafe {
                core::arch::asm!("in al, dx", out("al") lsr, in("dx") 0x3fdu16, options(nostack, preserves_flags));
            }
            if (lsr & 0x01) != 0 {
                let ch: u8;
                unsafe {
                    core::arch::asm!("in al, dx", out("al") ch, in("dx") 0x3f8u16, options(nostack, preserves_flags));
                }
                // kernel::contract!("[SERIAL] Received: 0x{:02x} ('{}')", ch, ch as char);
                buf.push(ch);
            } else {
                break;
            }
        }
    }

    fn init_uart(&self) {
        unsafe {
            let port = 0x3f8u16;
            // 1. Disable all interrupts while initializing
            core::arch::asm!("out dx, al", in("dx") port + 1, in("al") 0u8);

            // 2. Enable DLAB (set baud rate divisor)
            core::arch::asm!("out dx, al", in("dx") port + 3, in("al") 0x80u8);

            // 3. Set divisor to 1 (lo byte) 115200 baud
            core::arch::asm!("out dx, al", in("dx") port, in("al") 0x01u8);

            // 4. Set divisor to 0 (hi byte)
            core::arch::asm!("out dx, al", in("dx") port + 1, in("al") 0x00u8);

            // 5. 8 bits, no parity, one stop bit (clears DLAB)
            core::arch::asm!("out dx, al", in("dx") port + 3, in("al") 0x03u8);

            // 6. Enable FIFO, clear them, with 1-byte threshold
            core::arch::asm!("out dx, al", in("dx") port + 2, in("al") 0x07u8);

            // 7. IRQs enabled, RTS/DTR set
            // Bit 3 = OUT2. Bit 1 = RTS. Bit 0 = DTR.
            // 0x0B = 1011b (OUT2, RTS, DTR)
            // CRITICAL: OUT2 must be set to route interrupts to the PIC/IOAPIC!
            core::arch::asm!("out dx, al", in("dx") port + 4, in("al") 0x0Bu8);

            // 8. Re-enable interrupts: Received Data Available
            core::arch::asm!("out dx, al", in("dx") port + 1, in("al") 0x01u8);
        }
        kernel::contract!("[SERIAL] UART COM1 initialized (115200 8N1 FIFO-1 IRQ-on)");
    }
}

const BOOT_TEMP_MAP_BASE: u64 = 0xffffff10_00000000;

impl X86_64Runtime {
    pub const fn new() -> Self {
        Self {
            clamp: MonotonicClamp::new(),
            pci_root: AtomicU64::new(0),
            hhdm_offset: AtomicU64::new(0),
            cpu_ids: OnceCell::new(),
            timer_vector: AtomicUsize::new(0),
            timer_init_cnt: AtomicUsize::new(0),
            started_cpu_count: AtomicUsize::new(1),
            trampoline_ready: AtomicUsize::new(0),
            serial_buf: spin::Mutex::new(SerialBuffer::new()),
        }
    }

    /// TSC-based microsecond delay for SMP bring-up timing
    fn delay_us(&self, us: u64) {
        let ticks_to_wait = self.ticks_for_us(us);
        let start = self.mono_ticks();
        while self.mono_ticks().wrapping_sub(start) < ticks_to_wait {
            core::hint::spin_loop();
        }
    }

    #[inline]
    fn ticks_for_us(&self, us: u64) -> u64 {
        let freq = self.mono_freq_hz().max(1);
        // Round up so very short waits still make forward progress.
        freq.saturating_mul(us).saturating_add(999_999) / 1_000_000
    }

    fn wait_lapic_icr_idle(&self, lapic_virt: u64, timeout_us: u64) -> bool {
        let timeout_ticks = self.ticks_for_us(timeout_us).max(1);
        let start = self.mono_ticks();
        loop {
            let icr_low = unsafe { core::ptr::read_volatile((lapic_virt + 0x300) as *const u32) };
            if (icr_low & (1 << 12)) == 0 {
                return true;
            }
            if self.mono_ticks().wrapping_sub(start) >= timeout_ticks {
                return false;
            }
            core::hint::spin_loop();
        }
    }

    fn wait_for_ap_flag(&self, hhdm: u64, timeout_us: u64) -> bool {
        let timeout_ticks = self.ticks_for_us(timeout_us).max(1);
        let poll_interval_ticks = self.ticks_for_us(Self::AP_FLAG_POLL_INTERVAL_US).max(1);
        let start = self.mono_ticks();
        let mut next_poll = start;

        loop {
            let now = self.mono_ticks();
            if now.wrapping_sub(start) >= timeout_ticks {
                return false;
            }

            if now.wrapping_sub(next_poll) < poll_interval_ticks {
                core::hint::spin_loop();
                continue;
            }
            next_poll = now;

            let flag = unsafe { core::ptr::read_volatile((0x8500 + hhdm) as *const u64) };
            if flag == 1 {
                return true;
            }
        }
    }

    // Helper to map user entry pages
    fn map_user_entry(&self, entry: &UserEntry) -> Result<(), ()> {
        let page_mask = !(0xFFF_u64);
        let code_base = (entry.entry_pc as u64) & page_mask;
        // Stack grows down, so mapped page is below SP
        let stack_base = ((entry.user_sp as u64).saturating_sub(8)) & page_mask;

        let aspace = self.active_address_space();

        // 1. Map Code Page
        // Need to translate first to preserve physical address
        if let Some(code_phys) = self.translate(aspace, code_base) {
            self.map_page(
                aspace,
                code_base,
                code_phys,
                MapPerms {
                    user: true,
                    read: true,
                    write: false,
                    exec: true,
                    kind: MapKind::Normal,
                },
                MapKind::Normal,
                &ProxyAllocator,
            )?;
        } else {
            // If code is not mapped, we can't run!
            // But maybe we should return Err?
            // The caller unwraps.
            return Err(());
        }

        // 2. Map Stack Page
        if let Some(stack_phys) = self.translate(aspace, stack_base) {
            self.map_page(
                aspace,
                stack_base,
                stack_phys,
                MapPerms {
                    user: true,
                    read: true,
                    write: true,
                    exec: false,
                    kind: MapKind::Normal,
                },
                MapKind::Normal,
                &ProxyAllocator,
            )?;
        } else {
            return Err(());
        }

        Ok(())
    }
}

pub use paging::X86_64AddressSpace;
pub use task::X86_64Context;

impl ArchRuntime for X86_64Runtime {
    type Context = X86_64Context;
    type AddressSpace = X86_64AddressSpace;

    fn init(&self, hhdm_offset: u64) {
        self.hhdm_offset.store(hhdm_offset, Ordering::SeqCst);

        unsafe {
            // Initialize syscalls early (sets GS_BASE) so current_cpu_index() works
            syscall::init(0);

            gdt::init();

            // Allocate Double Fault Stack
            let phys = kernel::memory::alloc_frame().expect("No frames for DF stack");
            let virt = phys + hhdm_offset + 4096; // Top of stack
            gdt::set_ist1(virt);

            idt::init();
        }
        paging::init(hhdm_offset);

        // Map the LAPIC MMIO region into the HHDM.
        // The LAPIC is at 0xfee00000 and is not part of the normal memory map,
        // so we need to map it explicitly.
        let lapic_phys = 0xfee00000u64;
        let lapic_virt = lapic_phys + hhdm_offset;
        let aspace = self.active_address_space();
        // Check if already mapped (it might be if Limine's HHDM covers it)
        if paging::try_translate(aspace, lapic_virt).is_none() {
            // Create a simple frame hook for the mapping
            struct LocalFrameHook;
            impl FrameAllocatorHook for LocalFrameHook {
                fn alloc_frame(&self) -> Option<u64> {
                    kernel::memory::alloc_frame()
                }
            }

            // Map as uncacheable device memory
            let perms = kernel::MapPerms {
                read: true,
                write: true,
                exec: false,
                user: false,
                kind: kernel::MapKind::Device,
            };
            let _ = paging::map_page(
                aspace,
                lapic_virt,
                lapic_phys,
                perms,
                kernel::MapKind::Device,
                &LocalFrameHook,
            );
            paging::tlb_flush_page(lapic_virt);
        }

        // Initialize IOAPIC for interrupt routing (after IDT is set up)
        crate::arch::init_ioapic();

        // Perform full UART initialization (baud, MCR OUT2, etc)
        self.init_uart();
    }

    fn putchar(&self, c: u8) {
        unsafe {
            let port = 0x3f8u16;
            // Wait for Transmitter Holding Register Empty (THRE)
            loop {
                let lsr: u8;
                core::arch::asm!("in al, dx", out("al") lsr, in("dx") port + 5, options(nostack, preserves_flags));
                if (lsr & 0x20) != 0 {
                    break;
                }
                core::hint::spin_loop();
            }
            core::arch::asm!("out dx, al", in("dx") port, in("al") c);
        }
    }

    fn getchar(&self) -> Option<u8> {
        let irq = self.irq_disable();
        let res = {
            let mut buf = self.serial_buf.lock();
            if let Some(c) = buf.pop() {
                Some(c)
            } else {
                // Otherwise poll hardware
                self.poll_serial_unlocked(&mut buf);
                buf.pop()
            }
        };
        self.irq_restore(irq);
        res
    }

    fn halt(&self) -> ! {
        hcf()
    }

    fn reboot(&self) -> ! {
        kernel::kinfo!("X86_64: performing reboot via 8042");
        // Pulse the CPU reset line via the 8042 keyboard controller
        unsafe {
            // Wait for the keyboard controller input buffer to be clear
            let mut attempts = 0u32;
            while attempts < 100_000 {
                let status: u8;
                core::arch::asm!("in al, dx", out("al") status, in("dx") 0x64u16, options(nostack, preserves_flags));
                if status & 0x02 == 0 {
                    break;
                }
                attempts += 1;
            }
            // Send the reset command (0xFE = pulse reset line)
            core::arch::asm!("out dx, al", in("dx") 0x64u16, in("al") 0xFEu8, options(nostack, preserves_flags));
        }

        // Fallback 1: Triple Fault
        kernel::kinfo!("X86_64: reboot failed, attempting triple fault");
        unsafe {
            core::arch::asm!("lidt [{}]", in(reg) 0, options(nostack, preserves_flags));
            core::arch::asm!("int3", options(noreturn));
        } 

        // If that didn't work, halt
        hcf()
    }

    fn shutdown(&self) -> ! {
        kernel::kinfo!("X86_64: performing shutdown (QEMU/ACPI)");
        
        // 1. QEMU shutdown (newer)
        unsafe {
            core::arch::asm!("out dx, ax", in("dx") 0x604u16, in("ax") 0x2000u16, options(nostack, preserves_flags));
        }

        // 2. QEMU shutdown (older)
        unsafe {
            core::arch::asm!("out dx, ax", in("dx") 0xB004u16, in("ax") 0x2000u16, options(nostack, preserves_flags));
        }

        // 3. VirtualBox / Bochs shutdown
        unsafe {
            core::arch::asm!("out dx, ax", in("dx") 0x4004u16, in("ax") 0x3400u16, options(nostack, preserves_flags));
        }

        kernel::kinfo!("X86_64: shutdown failed, halting");
        hcf()
    }

    fn wait_for_interrupt(&self) {
        // Enable interrupts and halt until next IRQ
        unsafe {
            core::arch::asm!("sti", "hlt", options(nomem, nostack));
        }
    }

    fn mono_ticks(&self) -> u64 {
        let low: u32;
        let high: u32;
        unsafe {
            core::arch::asm!("rdtsc", out("eax") low, out("edx") high);
        }
        let raw = ((high as u64) << 32) | (low as u64);
        self.clamp.clamp(raw)
    }

    fn mono_freq_hz(&self) -> u64 {
        2_000_000_000
    }

    fn irq_disable(&self) -> IrqState {
        let rflags: u64;
        unsafe {
            core::arch::asm!("pushfq", "pop {}", out(reg) rflags);
            core::arch::asm!("cli");
        }
        IrqState(((rflags >> 9) & 1) as usize)
    }

    fn irq_restore(&self, state: IrqState) {
        if state.0 != 0 {
            unsafe {
                core::arch::asm!("sti");
            }
        }
    }

    fn threads_supported(&self) -> bool {
        true
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

    fn fence_full(&self) {
        unsafe {
            core::arch::asm!("mfence", options(nostack, preserves_flags));
        }
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
    unsafe fn switch(&self, from: &mut Self::Context, to: &Self::Context, to_tid: u64) {
        unsafe { task::switch(from, to, to_tid) }
    }

    unsafe fn switch_with_tls(
        &self,
        from: &mut Self::Context,
        to: &Self::Context,
        to_tid: u64,
        from_user_fs_base: *mut u64,
        to_user_fs_base: u64,
    ) {
        unsafe {
            // Save the outgoing thread's user FS_BASE into its task record.
            *from_user_fs_base = tls::read_user_fs_base();
            // Restore the incoming thread's user FS_BASE.
            tls::write_user_fs_base(to_user_fs_base);
            // Perform the actual register/stack context switch.
            task::switch(from, to, to_tid);
        }
    }

    fn get_user_tls_base(&self) -> u64 {
        tls::read_user_fs_base()
    }

    fn set_user_tls_base(&self, base: u64) {
        unsafe { tls::write_user_fs_base(base) }
    }

    unsafe fn enter_user(&self, entry: UserEntry) -> ! {
        // Map required pages before entering user mode
        let tid = self.current_tid();
        kernel::kdebug!(
            "ENTER_USER: TID={} entry_pc={:x} user_sp={:x}",
            tid,
            entry.entry_pc,
            entry.user_sp
        );

        self.map_user_entry(&entry)
            .expect("failed to map user entry pages");
        // x86_64 user mode entry via IRETQ
        let user_data_sel: u64 = gdt::USER_DATA_SEL as u64;
        let user_code_sel: u64 = gdt::USER_CODE_SEL as u64;
        let rflags: u64 = 0x202; // IF + reserved
        #[cfg(debug_assertions)]
        /*
        {
            let tid = unsafe { kernel::sched::current_tid_current() };
            let (cs, ss, rsp, rip, rflags_before, cr3, fs_base, gs_base) = capture_entry_state();
            let cpl = (cs & 0x3) as u64;
            kernel::log_event!(
                kernel::logging::LogLevel::Info,
                "bran::arch::x86_64::enter_user",
                "Entering user mode",
                {
                    tid: tid,
                    target_pc: entry.entry_pc as u64,
                    target_sp: entry.user_sp as u64,
                    target_cs: user_code_sel,
                    target_ss: user_data_sel,
                    CS: cs as u64,
                    SS: ss as u64,
                    CPL_KERNEL_BEFORE: cpl,
                    RIP_BEFORE: rip,
                    RSP_BEFORE: rsp,
                    RFLAGS_BEFORE: rflags_before,
                    CR3_BEFORE: cr3,
                    fs_base: fs_base,
                    gs_base: gs_base
                },
                about=[]
            );
        }
        */
        unsafe {
            asm!(
                "cli",
                "swapgs",
                "push {ss}",
                "push {rsp}",
                "push {rflags}",
                "push {cs}",
                "push {rip}",
                "iretq",
                ss = in(reg) user_data_sel,
                rsp = in(reg) entry.user_sp,
                rflags = in(reg) rflags,
                cs = in(reg) user_code_sel,
                rip = in(reg) entry.entry_pc,
                in("rdi") entry.arg0,
                options(noreturn)
            );
        }
    }

    // Paging
    fn make_user_address_space(&self) -> Self::AddressSpace {
        paging::make_user_address_space(self.active_address_space(), &ProxyAllocator)
    }
    fn active_address_space(&self) -> Self::AddressSpace {
        paging::active_address_space()
    }
    fn activate_address_space(&self, aspace: Self::AddressSpace) {
        unsafe {
            core::arch::asm!("mov cr3, {}", in(reg) aspace.0);
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
    fn protect_page(
        &self,
        aspace: Self::AddressSpace,
        virt: u64,
        perms: MapPerms,
    ) -> Result<(), ()> {
        paging::protect_page(aspace, virt, perms)
    }
    fn tlb_flush_page(&self, virt: u64) {
        paging::tlb_flush_page(virt)
    }

    fn aspace_to_raw(&self, aspace: Self::AddressSpace) -> u64 {
        aspace.0
    }

    fn setup_preemption_timer(&self, hz: u32) {
        let (init_cnt, ticks_per_sec) = ioapic::calibrate_lapic_timer(hz);

        self.timer_vector
            .store(idt::IRQ_TIMER_VECTOR as usize, Ordering::SeqCst);
        self.timer_init_cnt
            .store(init_cnt as usize, Ordering::SeqCst);

        ioapic::set_lapic_timer_periodic(idt::IRQ_TIMER_VECTOR, init_cnt);

        kernel::kdebug!(
            "LAPIC: calibrated timer ({} ticks/sec), init_cnt={} for {}Hz",
            ticks_per_sec,
            init_cnt,
            hz
        );
    }

    fn debug_active_aspace_root(&self) -> u64 {
        paging::active_address_space().0
    }

    fn pci_cfg_read32(
        &self,
        bus: u8,
        dev: u8,
        func: u8,
        offset: u8,
    ) -> Result<u32, abi::errors::Errno> {
        Ok(pci::read_config(bus, dev, func, offset))
    }
    fn pci_cfg_write32(
        &self,
        bus: u8,
        dev: u8,
        func: u8,
        offset: u8,
        value: u32,
    ) -> Result<(), abi::errors::Errno> {
        pci::write_config(bus, dev, func, offset, value);
        Ok(())
    }

    fn lapic_id(&self) -> Result<u32, abi::errors::Errno> {
        let hhdm = self.hhdm_offset.load(Ordering::SeqCst);
        // Safety check: hhdm must be non-zero (it's typically 0xffff800000000000 or similar)
        // If it's zero, the runtime isn't initialized yet and we can't read LAPIC safely
        if hhdm == 0 {
            return Err(abi::errors::Errno::EAGAIN);
        }
        let base = apic::base_phys();
        Ok(apic::id(base, hhdm))
    }
    fn lapic_base_phys(&self) -> Result<u64, abi::errors::Errno> {
        Ok(apic::base_phys())
    }

    fn map_phys_temp(&self, phys: u64, size: usize) -> Result<u64, abi::errors::Errno> {
        let aspace = self.active_address_space();
        let page_size = 4096u64;
        let phys_aligned = phys & !(page_size - 1);
        let offset = phys - phys_aligned;
        let size_to_map = (size as u64 + offset + (page_size - 1)) & !(page_size - 1);
        let pages = (size_to_map / page_size) as usize;

        for i in 0..pages {
            let p_addr = phys_aligned + (i as u64 * page_size);
            let v_addr = BOOT_TEMP_MAP_BASE + (i as u64 * page_size);
            self.map_page(
                aspace,
                v_addr,
                p_addr,
                MapPerms {
                    user: false,
                    read: true,
                    write: true, // Need write for trampoline copy
                    exec: false,
                    kind: MapKind::Normal,
                },
                MapKind::Normal,
                &ProxyAllocator,
            )
            .map_err(|_| abi::errors::Errno::ENOMEM)?;
            self.tlb_flush_page(v_addr);
        }

        Ok(BOOT_TEMP_MAP_BASE + offset)
    }

    fn cpu_ids(&self) -> &'static [CpuId] {
        let count = CPU_COUNT.load(Ordering::SeqCst) as usize;
        unsafe { &CPU_IDS[..count] }
    }

    fn next_offline_cpu(&self) -> Option<CpuId> {
        let started = self.started_cpu_count.load(Ordering::SeqCst);
        let total = CPU_COUNT.load(Ordering::SeqCst) as usize;
        if started < total {
            Some(unsafe { CPU_IDS[started] })
        } else {
            None
        }
    }

    fn current_cpu_id(&self) -> CpuId {
        match self.lapic_id() {
            Ok(id) => CpuId(id),
            Err(_) => CpuId(0),
        }
    }

    fn current_cpu_index(&self) -> usize {
        let idx: u64;
        unsafe {
            core::arch::asm!(
                "mov {}, gs:[16]",
                out(reg) idx,
                options(nostack, preserves_flags, readonly)
            );
        }
        idx as usize
    }

    fn current_tid(&self) -> u64 {
        let tid: u64;
        unsafe {
            core::arch::asm!(
                "mov {}, gs:[24]",
                out(reg) tid,
                options(nostack, preserves_flags, readonly)
            );
        }
        tid
    }

    fn set_current_tid(&self, tid: u64) {
        unsafe {
            core::arch::asm!(
                "mov gs:[24], {}",
                in(reg) tid,
                options(nostack, preserves_flags)
            );
        }
    }

    fn init_secondary_cpu(&self, cpu_index: usize) {
        // Load the kernel's GDT and IDT on this secondary CPU

        // The trampoline already set up CS=0x08 and DS/SS=0x10 with compatible descriptors.
        // We still need to load the kernel's GDT pointer so the TSS is available.
        unsafe {
            gdt::load_on_secondary(cpu_index);
        }

        // Load the IDT so we can handle exceptions
        unsafe {
            idt::load_on_secondary();
        }

        // Initialize syscalls for this CPU (sets GS_BASE)
        unsafe {
            syscall::init(cpu_index);
        }
        unsafe {
            idt::load_on_secondary();
        }

        // Enable the Local APIC on this secondary CPU.
        // The BSP enables its LAPIC in init_ioapic(), but secondary CPUs
        // need explicit enable too — without this, the SVR enable bit or
        // TPR may be in BIOS default state, silently masking IPIs like
        // the resched vector (0x30) and preventing task scheduling.
        ioapic::enable_local_apic();

        // Initialize preemption timer using cached BSP calibration
        let vector = self.timer_vector.load(Ordering::SeqCst);
        let init_cnt = self.timer_init_cnt.load(Ordering::SeqCst);
        if vector != 0 && init_cnt != 0 {
            ioapic::set_lapic_timer_periodic(vector as u8, init_cnt as u32);
        }
    }

    fn send_ipi(&self, cpu_index: usize, vector: u8) {
        // Read CPU_IDS/CPU_COUNT directly — don't use self.cpu_ids OnceCell
        // which is never initialized (the ArchRuntime trait's cpu_ids() method
        // at line 567 bypasses the inherent method that would initialize it).
        let count = CPU_COUNT.load(Ordering::SeqCst) as usize;
        if cpu_index < count {
            let apic_id = unsafe { CPU_IDS[cpu_index].0 };
            kernel::kdebug!(
                "SMP: send_ipi cpu_index={} apic_id={} vector=0x{:x}",
                cpu_index,
                apic_id,
                vector
            );
            ioapic::send_fixed_ipi(apic_id, vector);
        }
    }

    fn tlb_shootdown_broadcast(&self) {
        let current_cpu = self.current_cpu_index();
        let cpu_count = CPU_COUNT.load(Ordering::SeqCst) as usize;

        for i in 0..cpu_count {
            if i != current_cpu {
                self.send_ipi(i, idt::IRQ_TLB_SHOOTDOWN_VECTOR);
            }
        }
    }

    fn fill_entropy(&self, dst: &mut [u8]) -> usize {
        entropy::fill_entropy(dst)
    }

    unsafe fn start_cpu(
        &self,
        cpu: CpuId,
        entry: extern "C" fn(usize) -> !,
        cpu_index: usize,
    ) -> Result<(), abi::errors::Errno> {
        use core::sync::atomic::Ordering;
        use kernel::{MapKind, MapPerms, kdebug, kerror, kwarn};

        let hhdm = self.hhdm_offset.load(Ordering::SeqCst);
        let aspace = self.active_address_space();
        let trampoline_base: u64 = 0x8000;
        let trampoline_addr = trampoline_base + hhdm;

        // 1. One-time trampoline setup
        if self
            .trampoline_ready
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            kernel::kdebug!("SMP: Initializing trampoline at 0x{:x}", trampoline_base);

            self.map_page(
                aspace,
                trampoline_base,
                trampoline_base,
                MapPerms {
                    user: false,
                    read: true,
                    write: true,
                    exec: true,
                    kind: MapKind::Device,
                },
                MapKind::Device,
                &ProxyAllocator,
            )
            .map_err(|_| {
                kernel::kerror!("SMP: Failed to identity-map trampoline");
                abi::errors::Errno::ENOMEM
            })?;
            self.tlb_flush_page(trampoline_base);

            unsafe {
                let start = &smp::trampoline_start as *const _ as *const u8;
                let end = &smp::trampoline_end as *const _ as *const u8;
                let len = end.offset_from(start) as usize;

                if len > 4096 {
                    panic!("SMP: Trampoline too large!");
                }

                core::ptr::copy_nonoverlapping(start, trampoline_addr as *mut u8, len);
            }
            self.trampoline_ready.store(2, Ordering::SeqCst); // 2 means fully ready
        } else {
            // Wait for trampoline to be ready if another CPU is initializing it
            while self.trampoline_ready.load(Ordering::SeqCst) < 2 {
                core::hint::spin_loop();
            }
        }

        // 2. Start the specific AP
        let apic_id = cpu.0 as u32;
        kernel::kdebug!("SMP: Starting CPU {} (APIC {})", cpu_index, apic_id);

        let write_trampoline_data = |offset: usize, val: u64| unsafe {
            core::ptr::write_volatile((trampoline_addr + offset as u64) as *mut u64, val);
        };

        let lapic_base = self.lapic_base_phys().unwrap();
        let lapic_virt = lapic_base + hhdm;

        let write_icr = |high: u32, low: u32| unsafe {
            if !self.wait_lapic_icr_idle(lapic_virt, Self::LAPIC_ICR_DELIVERY_TIMEOUT_US) {
                kernel::kwarn!(
                    "SMP: LAPIC ICR busy before IPI to CPU {} (APIC {})",
                    cpu_index,
                    apic_id
                );
            }
            core::ptr::write_volatile((lapic_virt + 0x310) as *mut u32, high);
            core::ptr::write_volatile((lapic_virt + 0x300) as *mut u32, low);
            if !self.wait_lapic_icr_idle(lapic_virt, Self::LAPIC_ICR_DELIVERY_TIMEOUT_US) {
                kernel::kwarn!(
                    "SMP: LAPIC ICR still busy after IPI to CPU {} (APIC {})",
                    cpu_index,
                    apic_id
                );
            }
        };

        // 4 pages (16KB) for bootstrap stack
        let stack_frames = 4;
        let stack_base_phys = kernel::memory::alloc_contiguous_frames(stack_frames)
            .expect("Failed to alloc AP stack");
        let stack_top_virt = stack_base_phys + hhdm + (stack_frames as u64 * 4096);

        // Setup trampoline data
        write_trampoline_data(0x500, 0); // Clear flag
        write_trampoline_data(0x508, self.debug_active_aspace_root()); // CR3
        write_trampoline_data(0x510, stack_top_virt);
        write_trampoline_data(0x518, entry as usize as u64);
        write_trampoline_data(0x520, cpu_index as u64);
        write_trampoline_data(0x528, hhdm);

        // GDT descriptor at 0x530
        unsafe {
            let gdt_base = core::ptr::addr_of!(gdt::GDT_ARRAY[cpu_index]) as u64;
            let gdt_size = (core::mem::size_of::<gdt::Gdt>() - 1) as u16;
            let mut desc = [0u8; 10];
            desc[0..2].copy_from_slice(&gdt_size.to_le_bytes());
            desc[2..10].copy_from_slice(&gdt_base.to_le_bytes());
            for (off, &v) in desc.iter().enumerate() {
                core::ptr::write_volatile((trampoline_addr + 0x530 + off as u64) as *mut u8, v);
            }
        }

        // IDT descriptor at 0x540
        unsafe {
            let idt_base = core::ptr::addr_of!(idt::IDT) as u64;
            let idt_size = (core::mem::size_of::<idt::Idt>() - 1) as u16;
            let mut desc = [0u8; 10];
            desc[0..2].copy_from_slice(&idt_size.to_le_bytes());
            desc[2..10].copy_from_slice(&idt_base.to_le_bytes());
            for (off, &v) in desc.iter().enumerate() {
                core::ptr::write_volatile((trampoline_addr + 0x540 + off as u64) as *mut u8, v);
            }
        }

        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

        // INIT IPI
        write_icr(apic_id << 24, 0x0000C500);

        // Wait 10ms (Intel spec: 10ms after INIT before first SIPI)
        self.delay_us(10_000);

        // First SIPI
        write_icr(apic_id << 24, 0x00004608);
        // Wait 200µs (Intel spec: 200µs between SIPIs)
        self.delay_us(200);
        // Second SIPI (required by some CPUs)
        write_icr(apic_id << 24, 0x00004608);

        // Wait for come up (bounded)
        let came_up = self.wait_for_ap_flag(hhdm, Self::AP_STARTUP_TIMEOUT_US);

        if came_up {
            kernel::kdebug!("SMP: CPU {} (APIC {}) is online", cpu_index, apic_id);
            self.started_cpu_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        } else {
            kernel::kerror!("SMP: CPU {} (APIC {}) timed out", cpu_index, apic_id);
            Err(abi::errors::Errno::ETIMEDOUT)
        }
    }

    fn start_secondary_cpus(
        &self,
        entry: extern "C" fn(usize) -> !,
    ) -> Result<(), abi::errors::Errno> {
        use kernel::{kdebug, kerror};

        let total = CPU_COUNT.load(Ordering::SeqCst) as usize;
        if total <= 1 {
            kernel::kdebug!("SMP: No secondary CPUs to start");
            return Ok(());
        }

        kernel::kdebug!("SMP: Starting {} secondary CPUs...", total - 1);
        let mut first_err: Option<abi::errors::Errno> = None;

        for cpu_index in 1..total {
            let cpu_id = unsafe { CPU_IDS[cpu_index] };
            let result = unsafe { self.start_cpu(cpu_id, entry, cpu_index) };
            if let Err(err) = result {
                kernel::kerror!(
                    "SMP: Failed to start CPU {} (APIC {}): {:?}",
                    cpu_index,
                    cpu_id.0,
                    err
                );
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }

        if let Some(err) = first_err {
            Err(err)
        } else {
            kernel::kdebug!("SMP: Secondary CPU startup complete");
            Ok(())
        }
    }

    fn unmap_phys_temp(&self, virt: u64, size: usize) {
        let aspace = self.active_address_space();
        let page_size = 4096u64;
        let virt_aligned = virt & !(page_size - 1);
        let offset = virt - virt_aligned;
        let size_to_unmap = (size as u64 + offset + (page_size - 1)) & !(page_size - 1);
        let pages = (size_to_unmap / page_size) as usize;

        for i in 0..pages {
            let v_addr = virt_aligned + (i as u64 * page_size);
            let _ = self.unmap_page(aspace, v_addr);
            self.tlb_flush_page(v_addr);
        }
    }
}

struct ProxyAllocator;
impl FrameAllocatorHook for ProxyAllocator {
    fn alloc_frame(&self) -> Option<u64> {
        kernel::memory::alloc_frame()
    }
}

#[cfg(debug_assertions)]
fn capture_entry_state() -> (u16, u16, u64, u64, u64, u64, u64, u64) {
    let cs: u16;
    let ss: u16;
    let rsp: u64;
    let rip: u64;
    let rflags: u64;
    let cr3: u64;

    unsafe {
        core::arch::asm!(
            "mov {cs_out:x}, cs",
            "mov {ss_out:x}, ss",
            "mov {rsp_out}, rsp",
            "lea {rip_out}, [rip]",
            "pushfq",
            "pop {rflags_out}",
            cs_out = out(reg) cs,
            ss_out = out(reg) ss,
            rsp_out = out(reg) rsp,
            rip_out = out(reg) rip,
            rflags_out = out(reg) rflags,
        );
        core::arch::asm!("mov {cr3_out}, cr3", cr3_out = out(reg) cr3);
    }

    let fs_base = read_msr_debug(0xC000_0100);
    let gs_base = read_msr_debug(0xC000_0101);

    (cs, ss, rsp, rip, rflags, cr3, fs_base, gs_base)
}

#[cfg(debug_assertions)]
fn read_msr_debug(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") low,
            out("edx") high,
            options(nostack, preserves_flags)
        );
    }
    ((high as u64) << 32) | (low as u64)
}

pub fn hcf() -> ! {
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}
