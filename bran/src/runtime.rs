use kernel::{
    BootModuleDesc, BootRuntime, BootRuntimeBase, BootTasking, CpuId, FrameAllocatorHook,
    FramebufferInfo, IrqState, MapKind, MapPerms, PhysRange, UserEntry, UserTaskSpec,
};

pub trait ArchRuntime {
    type Context: Copy + Default;
    type AddressSpace: Copy + Default;

    fn init(&self, hhdm_offset: u64);
    fn putchar(&self, c: u8);
    /// Non-blocking serial read. Returns `Some(byte)` if data is available.
    fn getchar(&self) -> Option<u8> {
        None
    }
    fn halt(&self) -> !;
    fn shutdown(&self) -> ! {
        self.halt()
    }
    fn reboot(&self) -> ! {
        self.halt()
    }
    fn mono_ticks(&self) -> u64;
    fn mono_freq_hz(&self) -> u64;
    fn irq_disable(&self) -> IrqState;
    fn irq_restore(&self, state: IrqState);
    fn setup_preemption_timer(&self, _hz: u32) {}

    // SIMD - defaults
    fn simd_init_cpu(&self) {}
    fn simd_state_layout(&self) -> (usize, usize) {
        (0, 1)
    }
    unsafe fn simd_save(&self, _dst: *mut u8) {}
    unsafe fn simd_restore(&self, _src: *const u8) {}

    // Very early initialization (e.g., stack mode switching)
    unsafe fn early_init(&self) {}

    // Barriers - defaults
    fn threads_supported(&self) -> bool {
        false
    }
    fn fence_full(&self) {}
    fn icache_invalidate(&self) {}

    fn cpu_ids(&self) -> &'static [CpuId] {
        static ONE: [CpuId; 1] = [CpuId(0)];
        &ONE
    }
    fn current_cpu_id(&self) -> CpuId {
        CpuId(0)
    }

    fn cpu_total_count(&self) -> usize {
        self.cpu_ids().len()
    }
    fn next_offline_cpu(&self) -> Option<CpuId> {
        None
    }
    unsafe fn start_cpu(
        &self,
        _cpu: CpuId,
        _entry: extern "C" fn(usize) -> !,
        _arg: usize,
    ) -> Result<(), abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }

    /// Deprecated: use start_cpu for lazy bring-up.
    fn start_secondary_cpus(
        &self,
        _entry: extern "C" fn(usize) -> !,
    ) -> Result<(), abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }
    fn current_cpu_index(&self) -> usize {
        0
    }
    /// Per-CPU initialization for secondary cores.
    /// Called on each secondary CPU after it starts.
    fn init_secondary_cpu(&self, cpu_index: usize) {}

    /// Send an Inter-Processor Interrupt (IPI) to a specific CPU.
    fn send_ipi(&self, _cpu_index: usize, _vector: u8) {}
    fn tlb_shootdown_broadcast(&self) {}

    fn current_tid(&self) -> u64 {
        0
    }
    fn set_current_tid(&self, _tid: u64) {}

    // Wait for interrupt - low-power idle until next IRQ
    fn wait_for_interrupt(&self) {}


    // Tasking - defaults
    fn init_kernel_context(
        &self,
        _entry: extern "C" fn(usize) -> !,
        _stack_top: u64,
        _arg: usize,
    ) -> Self::Context {
        Self::Context::default()
    }
    fn init_user_context(
        &self,
        _spec: UserTaskSpec<Self::AddressSpace>,
        _kstack_top: u64,
    ) -> Self::Context {
        Self::Context::default()
    }
    unsafe fn switch(&self, _from: &mut Self::Context, _to: &Self::Context, _to_tid: u64) {
        // No-op
    }
    unsafe fn switch_with_tls(
        &self,
        from: &mut Self::Context,
        to: &Self::Context,
        to_tid: u64,
        _from_user_fs_base: *mut u64,
        _to_user_fs_base: u64,
    ) {
        unsafe { self.switch(from, to, to_tid) }
    }
    fn get_user_tls_base(&self) -> u64 {
        0
    }
    fn set_user_tls_base(&self, _base: u64) {}
    unsafe fn enter_user(&self, _entry: UserEntry) -> ! {
        panic!("enter_user not implemented for this architecture");
    }

    fn make_user_address_space(&self) -> Self::AddressSpace {
        Self::AddressSpace::default()
    }
    fn active_address_space(&self) -> Self::AddressSpace {
        Self::AddressSpace::default()
    }
    fn activate_address_space(&self, _aspace: Self::AddressSpace) {
        // No-op
    }

    fn map_page(
        &self,
        _aspace: Self::AddressSpace,
        _virt: u64,
        _phys: u64,
        _perms: MapPerms,
        _kind: MapKind,
        _allocator: &dyn FrameAllocatorHook,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn unmap_page(&self, _aspace: Self::AddressSpace, _virt: u64) -> Result<Option<u64>, ()> {
        Ok(None)
    }
    fn translate(&self, _aspace: Self::AddressSpace, _virt: u64) -> Option<u64> {
        None
    }
    fn protect_page(
        &self,
        _aspace: Self::AddressSpace,
        _virt: u64,
        _perms: MapPerms,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn tlb_flush_page(&self, _virt: u64) {}

    /// Convert an address-space handle to a raw `u64` token for storage in
    /// the architecture-independent [`kernel::task::Process`] struct.
    ///
    /// Default returns 0; override in architectures that maintain per-process
    /// page-table roots (all real hardware implementations should override).
    fn aspace_to_raw(&self, _aspace: Self::AddressSpace) -> u64 {
        0
    }

    // IO Port primitives (x86-only, stubs for other archs)
    fn ioport_read_u8(&self, _port: u16) -> u8 {
        0
    }
    fn ioport_read_u16(&self, _port: u16) -> u16 {
        0
    }
    fn ioport_read_u32(&self, _port: u16) -> u32 {
        0
    }
    fn ioport_write_u8(&self, _port: u16, _value: u8) {}
    fn ioport_write_u16(&self, _port: u16, _value: u16) {}
    fn ioport_write_u32(&self, _port: u16, _value: u32) {}

    fn debug_active_aspace_root(&self) -> u64 {
        0
    }

    fn pci_cfg_read32(
        &self,
        _bus: u8,
        _dev: u8,
        _func: u8,
        _offset: u8,
    ) -> Result<u32, abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }
    fn pci_cfg_write32(
        &self,
        _bus: u8,
        _dev: u8,
        _func: u8,
        _offset: u8,
        _value: u32,
    ) -> Result<(), abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }

    fn lapic_id(&self) -> Result<u32, abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }
    fn lapic_base_phys(&self) -> Result<u64, abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }

    fn map_phys_temp(&self, _phys: u64, _size: usize) -> Result<u64, abi::errors::Errno> {
        Err(abi::errors::Errno::NotSupported)
    }

    fn unmap_phys_temp(&self, _virt: u64, _size: usize) {}

    /// Fill buffer with hardware entropy bytes.
    /// Returns the number of bytes actually filled (0 = no HW RNG available).
    fn fill_entropy(&self, _dst: &mut [u8]) -> usize {
        0
    }
}

// --- Generic Runtime ---

pub struct Runtime<A: ArchRuntime> {
    pub arch: A,
    pub limine: LimineRuntimeData,
}

impl<A: ArchRuntime> Runtime<A> {
    pub const fn new(arch: A) -> Self {
        Self {
            arch,
            limine: LimineRuntimeData::new(),
        }
    }
}

pub struct LimineRuntimeData {}

impl LimineRuntimeData {
    pub const fn new() -> Self {
        Self {}
    }

    pub fn phys_memory_map(&self) -> &'static [PhysRange] {
        crate::mem::memory_map()
    }

    pub fn phys_to_virt_offset(&self) -> u64 {
        crate::requests::HHDM_REQUEST
            .get_response()
            .map(|r| r.offset())
            .unwrap_or(0)
    }

    pub fn modules(&self) -> &'static [BootModuleDesc] {
        crate::requests::get_modules()
    }

    pub fn framebuffer(&self) -> Option<FramebufferInfo> {
        crate::framebuffer::get_info()
    }

    pub fn acpi_rsdp(&self) -> Option<u64> {
        crate::requests::RSDP_REQUEST
            .get_response()
            .map(|r| r.address() as u64)
    }

    pub fn dtb_ptr(&self) -> Option<u64> {
        crate::requests::DTB_REQUEST
            .get_response()
            .map(|r| r.dtb_ptr() as u64)
    }
}

impl<A: ArchRuntime + 'static> BootRuntimeBase for Runtime<A> {
    fn putchar(&self, c: u8) {
        // Write to serial (arch-specific)
        self.arch.putchar(c);
        // Boot display path disabled; serial remains the early log sink.
        // crate::theme::putchar(c);
    }
    fn getchar(&self) -> Option<u8> {
        self.arch.getchar()
    }
    fn mono_ticks(&self) -> u64 {
        self.arch.mono_ticks()
    }
    fn mono_freq_hz(&self) -> u64 {
        self.arch.mono_freq_hz()
    }

    fn pci_cfg_read32(
        &self,
        bus: u8,
        dev: u8,
        func: u8,
        offset: u8,
    ) -> Result<u32, abi::errors::Errno> {
        self.arch.pci_cfg_read32(bus, dev, func, offset)
    }
    fn pci_cfg_write32(
        &self,
        bus: u8,
        dev: u8,
        func: u8,
        offset: u8,
        value: u32,
    ) -> Result<(), abi::errors::Errno> {
        self.arch.pci_cfg_write32(bus, dev, func, offset, value)
    }

    fn lapic_id(&self) -> Result<u32, abi::errors::Errno> {
        self.arch.lapic_id()
    }
    fn lapic_base_phys(&self) -> Result<u64, abi::errors::Errno> {
        self.arch.lapic_base_phys()
    }

    fn simd_init_cpu(&self) {
        self.arch.simd_init_cpu()
    }

    fn wait_for_interrupt(&self) {
        self.arch.wait_for_interrupt()
    }

    fn reboot(&self) -> ! {
        self.arch.reboot()
    }

    fn shutdown(&self) -> ! {
        self.arch.shutdown()
    }

    fn current_cpu_id(&self) -> CpuId {
        self.arch.current_cpu_id()
    }

    fn current_cpu_index(&self) -> usize {
        self.arch.current_cpu_index()
    }

    fn init_secondary_cpu(&self, cpu_index: usize) {
        self.arch.init_secondary_cpu(cpu_index)
    }

    fn cpu_total_count(&self) -> usize {
        self.arch.cpu_total_count()
    }

    fn next_offline_cpu(&self) -> Option<CpuId> {
        self.arch.next_offline_cpu()
    }

    unsafe fn start_cpu(
        &self,
        cpu: CpuId,
        entry: extern "C" fn(usize) -> !,
        arg: usize,
    ) -> Result<(), abi::errors::Errno> {
        unsafe { self.arch.start_cpu(cpu, entry, arg) }
    }

    fn send_ipi(&self, cpu_index: usize, vector: u8) {
        self.arch.send_ipi(cpu_index, vector)
    }

    fn current_tid(&self) -> u64 {
        self.arch.current_tid()
    }

    fn set_current_tid(&self, tid: u64) {
        self.arch.set_current_tid(tid)
    }

    fn tlb_shootdown_broadcast(&self) {
        self.arch.tlb_shootdown_broadcast()
    }

    fn fill_entropy(&self, dst: &mut [u8]) -> usize {
        self.arch.fill_entropy(dst)
    }

    fn phys_to_virt_offset(&self) -> u64 {
        self.limine.phys_to_virt_offset()
    }

    fn get_user_tls_base_dyn(&self) -> u64 {
        self.arch.get_user_tls_base()
    }

    fn set_user_tls_base_dyn(&self, base: u64) {
        self.arch.set_user_tls_base(base)
    }
}

impl<A: ArchRuntime + 'static> BootRuntime for Runtime<A> {
    type Tasking = Self;
    fn tasking(&self) -> &Self {
        self
    }

    fn halt(&self) -> ! {
        self.arch.halt()
    }

    // wait_for_interrupt moved to BootRuntimeBase

    // simd_init_cpu moved to BootRuntimeBase
    fn simd_state_layout(&self) -> (usize, usize) {
        self.arch.simd_state_layout()
    }
    unsafe fn simd_save(&self, dst: *mut u8) {
        unsafe { self.arch.simd_save(dst) }
    }
    unsafe fn simd_restore(&self, src: *const u8) {
        unsafe { self.arch.simd_restore(src) }
    }

    unsafe fn early_init(&self) {
        unsafe { self.arch.early_init() }
    }

    fn threads_supported(&self) -> bool {
        self.arch.threads_supported()
    }
    fn fence_full(&self) {
        self.arch.fence_full()
    }

    fn phys_memory_map(&self) -> &'static [PhysRange] {
        self.limine.phys_memory_map()
    }
    fn modules(&self) -> &'static [BootModuleDesc] {
        self.limine.modules()
    }
    fn framebuffer(&self) -> Option<FramebufferInfo> {
        self.limine.framebuffer()
    }

    fn cpu_ids(&self) -> &'static [CpuId] {
        self.arch.cpu_ids()
    }
    fn start_secondary_cpus(
        &self,
        entry: extern "C" fn(usize) -> !,
    ) -> Result<(), abi::errors::Errno> {
        self.arch.start_secondary_cpus(entry)
    }

    fn irq_disable(&self) -> IrqState {
        self.arch.irq_disable()
    }
    fn irq_restore(&self, state: IrqState) {
        self.arch.irq_restore(state)
    }

    fn acpi_rsdp(&self) -> Option<u64> {
        self.limine.acpi_rsdp()
    }
    fn dtb_ptr(&self) -> Option<u64> {
        self.limine.dtb_ptr()
    }

    // IO Port forwarding
    fn ioport_read_u8(&self, port: u16) -> u8 {
        self.arch.ioport_read_u8(port)
    }
    fn ioport_read_u16(&self, port: u16) -> u16 {
        self.arch.ioport_read_u16(port)
    }
    fn ioport_read_u32(&self, port: u16) -> u32 {
        self.arch.ioport_read_u32(port)
    }
    fn ioport_write_u8(&self, port: u16, value: u8) {
        self.arch.ioport_write_u8(port, value)
    }
    fn ioport_write_u16(&self, port: u16, value: u16) {
        self.arch.ioport_write_u16(port, value)
    }
    fn ioport_write_u32(&self, port: u16, value: u32) {
        self.arch.ioport_write_u32(port, value)
    }

    fn setup_preemption_timer(&self, hz: u32) {
        self.arch.setup_preemption_timer(hz)
    }

    fn debug_active_aspace_root(&self) -> u64 {
        self.arch.debug_active_aspace_root()
    }

    fn map_phys_temp(&self, phys: u64, size: usize) -> Result<u64, abi::errors::Errno> {
        self.arch.map_phys_temp(phys, size)
    }

    fn unmap_phys_temp(&self, virt: u64, size: usize) {
        self.arch.unmap_phys_temp(virt, size)
    }
}

impl<A: ArchRuntime + 'static> BootTasking for Runtime<A> {
    type Runtime = Self;
    type Context = A::Context;
    type AddressSpace = A::AddressSpace;

    fn init(&self, hhdm_offset: u64) {
        self.arch.init(hhdm_offset)
    }

    fn init_kernel_context(
        &self,
        entry: extern "C" fn(usize) -> !,
        stack_top: u64,
        arg: usize,
    ) -> Self::Context {
        self.arch.init_kernel_context(entry, stack_top, arg)
    }

    fn init_user_context(
        &self,
        spec: UserTaskSpec<Self::AddressSpace>,
        kstack_top: u64,
    ) -> Self::Context {
        self.arch.init_user_context(spec, kstack_top)
    }

    unsafe fn switch(&self, from: &mut Self::Context, to: &Self::Context, to_tid: u64) {
        unsafe { self.arch.switch(from, to, to_tid) }
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
            self.arch
                .switch_with_tls(from, to, to_tid, from_user_fs_base, to_user_fs_base)
        }
    }

    fn get_user_tls_base(&self) -> u64 {
        self.arch.get_user_tls_base()
    }

    fn set_user_tls_base(&self, base: u64) {
        self.arch.set_user_tls_base(base)
    }

    unsafe fn enter_user(&self, entry: UserEntry) -> ! {
        unsafe { self.arch.enter_user(entry) }
    }

    fn make_user_address_space(&self) -> Self::AddressSpace {
        self.arch.make_user_address_space()
    }

    fn active_address_space(&self) -> Self::AddressSpace {
        self.arch.active_address_space()
    }

    fn activate_address_space(&self, aspace: Self::AddressSpace) {
        self.arch.activate_address_space(aspace)
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
        self.arch
            .map_page(aspace, virt, phys, perms, kind, allocator)
    }

    fn unmap_page(&self, aspace: Self::AddressSpace, virt: u64) -> Result<Option<u64>, ()> {
        self.arch.unmap_page(aspace, virt)
    }

    fn translate(&self, aspace: Self::AddressSpace, virt: u64) -> Option<u64> {
        self.arch.translate(aspace, virt)
    }

    fn protect_page(
        &self,
        aspace: Self::AddressSpace,
        virt: u64,
        perms: MapPerms,
    ) -> Result<(), ()> {
        self.arch.protect_page(aspace, virt, perms)
    }

    fn tlb_flush_page(&self, virt: u64) {
        self.arch.tlb_flush_page(virt)
    }

    fn aspace_to_raw(&self, aspace: Self::AddressSpace) -> u64 {
        self.arch.aspace_to_raw(aspace)
    }
}
