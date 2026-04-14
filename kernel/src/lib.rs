#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod device_registry;
pub mod entropy;
pub mod group;
pub mod ipc;
pub mod irq;
pub mod job;
pub mod logging;
pub mod memory;
pub mod once_cell;
pub mod signal;

pub mod sched;
pub mod simd;
pub mod syscall;
pub mod task;
pub mod vfs;

pub mod time;
pub mod trace;
pub mod virtio;

use crate::task::StartupArg;
use abi::errors::Errno;
use abi::vm::{VmBackingKind, VmMapFlags, VmProt, VmRegionInfo};

#[unsafe(no_mangle)]
pub extern "C" fn kernel_handle_page_fault(rip: u64, addr: u64, err: u64) {
    // Decode x86_64 page fault error code bits
    let present = (err & 0x1) != 0;
    let write = (err & 0x2) != 0;
    let user = (err & 0x4) != 0;
    let instr_fetch = (err & 0x10) != 0;

    let stack_result = if user {
        unsafe { crate::sched::handle_user_stack_fault_current(addr) }
    } else {
        crate::sched::StackFaultResult::NotStack
    };
    if stack_result == crate::sched::StackFaultResult::Grew {
        return;
    }

    let tid = unsafe { crate::sched::current_tid_current() };
    let name_bytes = unsafe { crate::sched::current_task_name_current() };
    let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(32);
    let task_name = core::str::from_utf8(&name_bytes[..name_len]).unwrap_or("unknown");

    // Structured page fault logging with decoded error bits
    crate::log_event!(
        crate::logging::LogLevel::Error,
        "kernel::trap",
        "user_page_fault tid={} task='{}' va=0x{:016x} rip=0x{:016x} err=0x{:04x} p={} u={} w={} i={}",
        tid,
        task_name,
        addr,
        rip,
        err,
        present as u8,
        user as u8,
        write as u8,
        instr_fetch as u8
    );

    if stack_result == crate::sched::StackFaultResult::Overflow {
        crate::kprintln!("STACK: overflow at va=0x{:x} (task {})", addr, task_name);
    }

    unsafe {
        crate::sched::exit_current(-1);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kernel_handle_exception(rip: u64, error_code: u64, rsp: u64, cs: u64, kind: u64) {
    let exception_name = match kind {
        0 => "Divide-by-zero",
        1 => "Debug",
        2 => "NMI",
        3 => "Breakpoint",
        4 => "Overflow",
        5 => "Bound Range",
        6 => "Invalid Opcode (UD2)",
        7 => "Device Not Available",
        8 => "Double Fault",
        10 => "Invalid TSS",
        11 => "Segment Not Present",
        12 => "Stack-Segment Fault",
        13 => "General Protection Fault",
        14 => "Page Fault",
        16 => "FPU Exception",
        17 => "Alignment Check",
        18 => "Machine Check",
        19 => "SIMD Exception",
        20 => "Virtualization",
        21 => "Control Protection",
        30 => "Security Exception",
        _ => "Unknown Exception",
    };

    let tid = unsafe { crate::sched::current_tid_current() };
    let name_bytes = unsafe { crate::sched::current_task_name_current() };
    let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(32);
    let task_name = core::str::from_utf8(&name_bytes[..name_len]).unwrap_or("unknown");

    // Peeking at instruction bytes for Invalid Opcode
    let mut instr_bytes = [0u8; 8];
    let mut peek_ok = false;
    if kind == 6 {
        for i in 0..8 {
            if let Some(phys) = crate::memory::translate_user_page(rip + i as u64) {
                let off = crate::runtime_base().phys_to_virt_offset();
                let ptr = (phys + off) as *const u8;
                unsafe {
                    instr_bytes[i] = *ptr;
                }
                peek_ok = true;
            } else {
                break;
            }
        }
    }

    if peek_ok {
        crate::log_event!(
            crate::logging::LogLevel::Error,
            "kernel::trap",
            "{} tid={} task='{}' rip=0x{:016x} rsp=0x{:016x} err=0x{:04x} instr={:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            exception_name,
            tid,
            task_name,
            rip,
            rsp,
            error_code,
            instr_bytes[0],
            instr_bytes[1],
            instr_bytes[2],
            instr_bytes[3],
            instr_bytes[4],
            instr_bytes[5],
            instr_bytes[6],
            instr_bytes[7]
        );
    } else {
        crate::log_event!(
            crate::logging::LogLevel::Error,
            "kernel::trap",
            "{} tid={} task='{}' rip=0x{:016x} rsp=0x{:016x} err=0x{:04x} kind={}",
            exception_name,
            tid,
            task_name,
            rip,
            rsp,
            error_code,
            kind
        );
    }

    unsafe {
        crate::sched::exit_current(-1);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PhysRange {
    pub start: u64,
    pub end: u64,
    pub kind: PhysRangeKind,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysRangeKind {
    Usable,
    Reserved,
    Mmio,
    Firmware,
    KernelImage,
    BootModule,
    Framebuffer,
    Acpi,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct BootModuleDesc {
    pub name: &'static str,
    pub cmdline: &'static str,
    pub bytes: &'static [u8],
    pub phys_start: u64,
    pub phys_end: u64,
    pub kind: BootModuleKind,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootModuleKind {
    Unknown,
    Elf,
    Wasm,
    Data,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FramebufferInfo {
    pub addr: u64,
    pub byte_len: u64,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u16,
    pub format: PixelFormat,
}

/// Pixel format for framebuffer surfaces.
///
/// Values match `abi::schema::pixel_format` constants for wire compatibility.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Unknown or unsupported format.
    Unknown = 0,
    /// 32-bit BGRA: Memory [B, G, R, A] -> u32 0xAARRGGBB.
    Bgra8888 = 1,
    /// 32-bit BGRX: Memory [B, G, R, X] -> u32 0xXXRRGGBB (alpha ignored).
    Bgrx8888 = 2,
    /// 16-bit RGB565.
    Rgb565 = 3,
}

impl PixelFormat {
    /// Convert to wire-compatible u64 for graph properties.
    #[inline]
    pub const fn to_wire(self) -> u64 {
        self as u64
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct IrqState(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CpuId(pub u32);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MapPerms {
    pub user: bool,
    pub read: bool,
    pub write: bool,
    pub exec: bool,
    pub kind: MapKind,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MapKind {
    #[default]
    Normal,
    Device,
    Framebuffer,
}

pub struct UserTaskSpec<AS> {
    pub entry: u64,
    pub stack_top: u64,
    pub aspace: AS,
    pub arg: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UserEntry {
    pub entry_pc: usize,
    pub user_sp: usize,
    pub arg0: usize,
}

pub trait FrameAllocatorHook {
    fn alloc_frame(&self) -> Option<u64>;
}

pub trait BootTasking {
    type Runtime: BootRuntime<Tasking = Self>;
    type Context: Copy + Default;
    type AddressSpace: Copy + Default;

    fn init(&self, hhdm_offset: u64);
    fn init_kernel_context(
        &self,
        entry: extern "C" fn(usize) -> !,
        stack_top: u64,
        arg: usize,
    ) -> Self::Context;
    fn init_user_context(
        &self,
        spec: UserTaskSpec<Self::AddressSpace>,
        kstack_top: u64,
    ) -> Self::Context;

    unsafe fn switch(&self, from: &mut Self::Context, to: &Self::Context, to_tid: u64);

    /// Like [`switch`] but also saves/restores the per-thread user TLS base.
    ///
    /// `from_user_fs_base` is a pointer into the outgoing task's `user_fs_base`
    /// field; on switch-out the arch layer reads the live hardware base and stores
    /// it there.  `to_user_fs_base` is the value loaded from the incoming task's
    /// field and written to hardware on switch-in.
    ///
    /// The default implementation simply delegates to [`switch`] — architectures
    /// without a dedicated user TLS register need no changes.
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

    /// Read the current thread's user TLS base from hardware (FS_BASE on x86_64).
    /// Returns 0 on architectures without a dedicated user TLS register.
    fn get_user_tls_base(&self) -> u64 {
        0
    }

    /// Write a new user TLS base to hardware immediately (FS_BASE on x86_64).
    /// No-op on architectures without a dedicated user TLS register.
    fn set_user_tls_base(&self, _base: u64) {}
    unsafe fn enter_user(&self, entry: UserEntry) -> !;

    fn make_user_address_space(&self) -> Self::AddressSpace;
    fn active_address_space(&self) -> Self::AddressSpace;
    fn activate_address_space(&self, aspace: Self::AddressSpace);

    fn map_page(
        &self,
        aspace: Self::AddressSpace,
        virt: u64,
        phys: u64,
        perms: MapPerms,
        kind: MapKind,
        allocator: &dyn FrameAllocatorHook,
    ) -> Result<(), ()>;

    fn unmap_page(&self, aspace: Self::AddressSpace, virt: u64) -> Result<Option<u64>, ()>;
    fn protect_page(
        &self,
        aspace: Self::AddressSpace,
        virt: u64,
        perms: MapPerms,
    ) -> Result<(), ()>;
    fn translate(&self, aspace: Self::AddressSpace, virt: u64) -> Option<u64>;
    fn tlb_flush_page(&self, virt: u64);

    /// Convert an address space handle to a raw `u64` token suitable for
    /// storage in the architecture-independent [`crate::task::Process`] struct.
    ///
    /// For single-register architectures (x86-64 CR3, RISC-V SATP, AArch64
    /// TTBR0) this returns the register value directly.  For dual-register
    /// architectures (LoongArch64 PGDL/PGDH) this returns the user-half
    /// register (PGDL); the kernel-half is shared across all processes and
    /// does not need per-process storage.
    ///
    /// The default implementation returns 0; architectures that require
    /// process-scoped address-space tracking must override this.
    fn aspace_to_raw(&self, _aspace: Self::AddressSpace) -> u64 {
        0
    }
}

pub trait BootRuntimeBase: 'static {
    fn putchar(&self, c: u8);
    /// Non-blocking serial read. Returns `Some(byte)` if data is available.
    fn getchar(&self) -> Option<u8> {
        None
    }
    fn mono_ticks(&self) -> u64;
    fn mono_freq_hz(&self) -> u64 {
        10_000_000
    }

    fn pci_cfg_read32(&self, _bus: u8, _dev: u8, _func: u8, _offset: u8) -> Result<u32, Errno> {
        Err(Errno::NotSupported)
    }
    fn pci_cfg_write32(
        &self,
        _bus: u8,
        _dev: u8,
        _func: u8,
        _offset: u8,
        _value: u32,
    ) -> Result<(), Errno> {
        Err(Errno::NotSupported)
    }

    fn lapic_id(&self) -> Result<u32, Errno> {
        Err(Errno::NotSupported)
    }
    fn lapic_base_phys(&self) -> Result<u64, Errno> {
        Err(Errno::NotSupported)
    }

    fn simd_init_cpu(&self) {}

    /// Wait for interrupt - low-power idle until next IRQ
    fn wait_for_interrupt(&self) {}

    /// Reboot the system. This should never return.
    fn reboot(&self) -> ! {
        loop {
            core::hint::spin_loop();
        }
    }

    /// Shutdown the system. This should never return.
    fn shutdown(&self) -> ! {
        loop {
            core::hint::spin_loop();
        }
    }

    /// Send an Inter-Processor Interrupt (IPI) to a specific CPU.
    fn send_ipi(&self, _cpu_index: usize, _vector: u8) {}

    fn current_cpu_id(&self) -> CpuId {
        CpuId(0)
    }

    fn current_cpu_index(&self) -> usize {
        0
    }

    fn current_tid(&self) -> u64 {
        0
    }

    fn set_current_tid(&self, _tid: u64) {}

    /// Broadcast a TLB shootdown IPI to all other CPUs.
    fn tlb_shootdown_broadcast(&self) {}

    /// Per-CPU initialization for secondary CPUs.
    /// Initialize a secondary CPU after it has entered the kernel.
    fn init_secondary_cpu(&self, cpu_index: usize);

    /// Total CPUs discovered on this platform.
    fn cpu_total_count(&self) -> usize {
        1
    }

    /// Returns the next offline CPU id.
    fn next_offline_cpu(&self) -> Option<CpuId> {
        None
    }

    /// Request that one CPU be started.
    unsafe fn start_cpu(
        &self,
        _cpu: CpuId,
        _entry: extern "C" fn(usize) -> !,
        _arg: usize,
    ) -> Result<(), Errno> {
        Err(Errno::NotSupported)
    }

    /// Fill buffer with hardware entropy bytes.
    /// Returns the number of bytes actually filled (0 = no HW RNG available).
    fn fill_entropy(&self, _dst: &mut [u8]) -> usize {
        0
    }

    fn phys_to_virt_offset(&self) -> u64;

    /// Read the current thread's user TLS base from hardware (FS_BASE on x86_64).
    /// Returns 0 on architectures without a dedicated user TLS register.
    fn get_user_tls_base_dyn(&self) -> u64 {
        0
    }

    /// Write a new user TLS base to hardware immediately (FS_BASE on x86_64).
    /// No-op on architectures without a dedicated user TLS register.
    fn set_user_tls_base_dyn(&self, _base: u64) {}
}

pub trait BootRuntime: BootRuntimeBase + Sized + 'static {
    type Tasking: BootTasking<Runtime = Self>;
    fn tasking(&self) -> &Self::Tasking;

    fn halt(&self) -> !;

    fn threads_supported(&self) -> bool {
        false
    }
    // simd_init_cpu moved to BootRuntimeBase
    fn simd_state_layout(&self) -> (usize, usize) {
        (0, 1)
    }
    unsafe fn simd_save(&self, _dst: *mut u8) {}
    unsafe fn simd_restore(&self, _src: *const u8) {}

    /// Very early architecture initialization, called before any significant stack usage.
    /// Used for critical setup like switching stack modes on AArch64.
    /// Default implementation does nothing.
    unsafe fn early_init(&self) {}

    fn fence_full(&self) {}
    fn icache_invalidate(&self) {}

    // wait_for_interrupt moved to BootRuntimeBase
    // phys_to_virt_offset moved to BootRuntimeBase

    fn phys_memory_map(&self) -> &'static [PhysRange];
    fn modules(&self) -> &'static [BootModuleDesc];
    fn framebuffer(&self) -> Option<FramebufferInfo>;

    fn page_size(&self) -> usize {
        4096
    }
    fn kernel_virt_base(&self) -> u64 {
        0xffffffff80000000
    }
    fn boot_cpu_id(&self) -> usize {
        0
    }
    fn cpu_ids(&self) -> &'static [CpuId] {
        const ONE: [CpuId; 1] = [CpuId(0)];
        &ONE
    }

    /// Start all non-boot CPUs and run `entry` on each of them.
    /// Deprecated: use start_cpu for lazy bring-up.
    fn start_secondary_cpus(&self, _entry: extern "C" fn(usize) -> !) -> Result<(), Errno> {
        Err(Errno::NotSupported)
    }

    fn irq_disable(&self) -> IrqState;
    fn irq_restore(&self, state: IrqState);

    /// Setup periodic preemption timer (e.g. 100Hz heartbeat)
    fn setup_preemption_timer(&self, _hz: u32) {}

    fn acpi_rsdp(&self) -> Option<u64> {
        None
    }
    fn dtb_ptr(&self) -> Option<u64> {
        None
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

    /// Map a physical range into a temporary virtual address for boot-time copies.
    /// This is used for reading firmware tables that might not be in the HHDM.
    /// Returns the virtual address of the start of the range.
    fn map_phys_temp(&self, _phys: u64, _size: usize) -> Result<u64, Errno> {
        Err(Errno::NotSupported)
    }

    /// Unmap a previously mapped temporary physical range.
    fn unmap_phys_temp(&self, _virt: u64, _size: usize) {}
}

// Per-CPU generic tracking
// In a full implementation, this should be a per-cpu structure or array.
// For now, we only trust this for the boot CPU or rely on atomic updates.
static CPU_ONLINE: once_cell::OnceCell<&'static core::sync::atomic::AtomicUsize> =
    once_cell::OnceCell::new();

static RUNTIME: once_cell::OnceCell<&'static dyn core::any::Any> = once_cell::OnceCell::new();
static RUNTIME_BASE: once_cell::OnceCell<&'static dyn BootRuntimeBase> = once_cell::OnceCell::new();
static mut RAW_RUNTIME_BASE: Option<&'static dyn BootRuntimeBase> = None;

/// Initialize the runtime. Panics if called more than once.
pub fn init_runtime<R: BootRuntime>(runtime: &'static R) {
    let name = core::any::type_name::<R>();
    RUNTIME.set(runtime as &'static dyn core::any::Any);
    RUNTIME_BASE.set(runtime as &'static dyn BootRuntimeBase);
    unsafe {
        RAW_RUNTIME_BASE = Some(runtime as &'static dyn BootRuntimeBase);
    }
    crate::contract!("INIT_RUNTIME: type={}", name);
}

pub fn runtime<R: BootRuntime>() -> &'static R {
    let any_ref: &'static dyn core::any::Any = *RUNTIME.get();
    if let Some(rt) = any_ref.downcast_ref::<R>() {
        rt
    } else {
        panic!(
            "Runtime type mismatch: expected {}",
            core::any::type_name::<R>()
        );
    }
}

pub fn runtime_base() -> &'static dyn BootRuntimeBase {
    *RUNTIME_BASE.get()
}

/// Returns `true` if the runtime has been initialized.
///
/// Safe to call from any context (including early boot and unit tests).
/// Code that cannot tolerate a panic on `runtime_base()` should guard with this.
pub fn is_runtime_initialized() -> bool {
    RUNTIME_BASE.is_initialized()
}

// Global IO port accessor functions
// On x86, these use inline asm. On other archs, they are no-ops.
#[inline]
pub fn ioport_read_u8(_port: u16) -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        let val: u8;
        unsafe {
            core::arch::asm!("in al, dx", out("al") val, in("dx") _port, options(nostack, preserves_flags))
        };
        val
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

#[inline]
pub fn ioport_read_u16(_port: u16) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        let val: u16;
        unsafe {
            core::arch::asm!("in ax, dx", out("ax") val, in("dx") _port, options(nostack, preserves_flags))
        };
        val
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

#[inline]
pub fn ioport_read_u32(_port: u16) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        let val: u32;
        unsafe {
            core::arch::asm!("in eax, dx", out("eax") val, in("dx") _port, options(nostack, preserves_flags))
        };
        val
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

#[inline]
pub fn ioport_write_u8(_port: u16, _val: u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("out dx, al", in("dx") _port, in("al") _val, options(nostack, preserves_flags))
    };
}

#[inline]
pub fn ioport_write_u16(_port: u16, _val: u16) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("out dx, ax", in("dx") _port, in("ax") _val, options(nostack, preserves_flags))
    };
}

#[inline]
pub fn ioport_write_u32(_port: u16, _val: u32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("out dx, eax", in("dx") _port, in("eax") _val, options(nostack, preserves_flags))
    };
}

struct GlobalAllocHook;
impl FrameAllocatorHook for GlobalAllocHook {
    fn alloc_frame(&self) -> Option<u64> {
        crate::memory::alloc_frame()
    }
}

fn _irq_disable_wrapper<R: BootRuntime>() -> IrqState {
    runtime::<R>().irq_disable()
}
fn _irq_restore_wrapper<R: BootRuntime>(state: IrqState) {
    runtime::<R>().irq_restore(state);
}

fn paint_bootfb_probe(fb: FramebufferInfo) {
    if fb.width == 0 || fb.height == 0 || fb.pitch < 4 || fb.byte_len < (fb.pitch as u64) {
        return;
    }

    let width = fb.width as usize;
    let height = fb.height as usize;
    let stride_px = (fb.pitch as usize) / 4;
    let rows = core::cmp::min(height, (fb.byte_len / fb.pitch as u64) as usize);

    let ptr = fb.addr as *mut u32;
    if ptr.is_null() || stride_px == 0 || rows == 0 {
        return;
    }

    unsafe {
        for y in 0..rows {
            let row = core::slice::from_raw_parts_mut(ptr.add(y * stride_px), width.min(stride_px));
            let band = (y * 4) / rows.max(1);
            let color = match band {
                0 => 0x00_30_30_A0,
                1 => 0x00_30_A0_30,
                2 => 0x00_A0_30_30,
                _ => 0x00_80_80_80,
            };
            for pixel in row.iter_mut() {
                *pixel = color;
            }
        }

        let marker_h = rows.min(96);
        let marker_w = width.min(stride_px).min(256);
        for y in 0..marker_h {
            let row = core::slice::from_raw_parts_mut(ptr.add(y * stride_px), width.min(stride_px));
            for x in 0..marker_w {
                row[x] = if ((x / 16) + (y / 16)) % 2 == 0 {
                    0x00_FF_FF_FF
                } else {
                    0x00_00_00_00
                };
            }
        }

        // Draw 'K' marker at (280, 10) to signal Kernel ownership
        let k_x = 280;
        let k_y = 10;
        if width > k_x + 60 && rows > k_y + 80 {
            for dy in 0..80 {
                let row = core::slice::from_raw_parts_mut(
                    ptr.add((k_y + dy) * stride_px),
                    width.min(stride_px),
                );
                // Vertical stem
                for dx in 0..12 {
                    if k_x + dx < row.len() {
                        row[k_x + dx] = 0x00_FF_FF_00;
                    }
                }
                // Diagonals
                let mid = 40;
                let arm_w = (dy as i32 - mid).abs();
                let dx = 12 + arm_w;
                for i in 0..12 {
                    if k_x + dx as usize + i < row.len() {
                        row[k_x + dx as usize + i] = 0x00_FF_FF_00;
                    }
                }
            }
        }
    }
}

pub fn start<R: BootRuntime>(runtime: &'static R) -> ! {
    crate::irq::IRQ_DISABLE_HOOK.store(
        _irq_disable_wrapper::<R> as *mut (),
        core::sync::atomic::Ordering::SeqCst,
    );
    crate::irq::IRQ_RESTORE_HOOK.store(
        _irq_restore_wrapper::<R> as *mut (),
        core::sync::atomic::Ordering::SeqCst,
    );

    init_runtime(runtime);
    unsafe { crate::logging::init(runtime) };

    if let Some(fb) = runtime.framebuffer() {
        crate::kdebug!(
            "BOOTFB: width={} height={} pitch={} bpp={} format={:?}",
            fb.width,
            fb.height,
            fb.pitch,
            fb.bpp,
            fb.format
        );
    }

    contract!("thing-os kernel starting...");

    memory::init(runtime);
    contract!("Initializing global allocator...");
    memory::global_alloc::init(runtime);

    if let Some(fb) = runtime.framebuffer() {
        let fb_resource_id = 0xFB00_0000;

        {
            let mut reg = crate::device_registry::REGISTRY.lock();
            let mut bars = [0; 6];
            let mut sizes = [0; 6];

            // CRITICAL: fb.addr from the runtime depends on the bootloader/arch,
            // but is typically a kernel virtual address (HHDM).
            // device_registry expects PHYSICAL addresses for BARs.
            let ph_offset = runtime.phys_to_virt_offset();
            let phys_addr = if fb.addr >= ph_offset {
                fb.addr - ph_offset
            } else {
                fb.addr
            };

            bars[0] = phys_addr;
            sizes[0] = fb.byte_len as u64;
            reg.register(crate::device_registry::DeviceEntry::new_mmio(
                "display_fb",
                fb_resource_id,
                bars,
                sizes,
            ));
        }

        crate::vfs::devfs::set_boot_fb(fb, fb_resource_id);
        crate::vfs::devfs::register(
            "fb0",
            alloc::sync::Arc::new(crate::vfs::devfs::FbNode::new(fb, fb_resource_id)),
        );
    }

    contract!("Seeding entropy pool...");
    crate::entropy::seed_from_hardware();

    contract!("Initializing SIMD...");
    runtime.simd_init_cpu();

    contract!("Initializing tasking...");
    crate::task::init::<R>();

    contract!("Initializing VFS...");
    crate::vfs::init(runtime.modules());

    kdebug!("Scanning PCI bus...");
    scan_pci();

    // CRITICAL: Calibrate the BSP preemption timer BEFORE starting secondary CPUs.
    // Secondary CPUs read timer_vector/timer_init_cnt in init_secondary_cpu().
    // If these aren't set yet, secondary CPUs get no LAPIC timer, meaning
    // wake_sleepers() (called only from on_tick → PreemptTick) never fires
    // on those CPUs, and any task that calls sleep_ms() is stuck forever.
    kdebug!("System initialized. Setting up preemption timer (100Hz)...");
    runtime.setup_preemption_timer(100);

    // Bring up all secondary CPUs during early boot.
    let cpu_total = runtime.cpu_total_count();
    if cpu_total > 1 {
        crate::kdebug!(
            "Kernel: Detected {} CPUs. Starting {} secondaries...",
            cpu_total,
            cpu_total - 1
        );
        match runtime.start_secondary_cpus(kernel_secondary_entry::<R>) {
            Ok(()) => crate::kdebug!("Kernel: Secondary CPU bring-up complete."),
            Err(err) => crate::kerror!("Kernel: Secondary CPU bring-up failed: {:?}", err),
        }
    } else {
        crate::kdebug!("Kernel: Detected {} CPU.", cpu_total);
    }

    // Store global boot info for syscalls
    crate::boot_info::set(crate::boot_info::BootSyscallInfo {
        memory_map: runtime.phys_memory_map(),
        modules: runtime.modules(),
        framebuffer: runtime.framebuffer(),
        hhdm_offset: runtime.phys_to_virt_offset(),
        acpi_rsdp: runtime.acpi_rsdp(),
        dtb_ptr: runtime.dtb_ptr(),
    });

    let modules = runtime.modules();
    kdebug!("Kernel: Enumerating {} boot modules...", modules.len());
    for (i, m) in modules.iter().enumerate() {
        crate::ktrace!(
            "  Module[{}]: name='{}' cmdline='{}' size={} bytes",
            i,
            m.name,
            m.cmdline,
            m.bytes.len()
        );
    }

    // Look for module with "init" in cmdline, otherwise fallback to "sprout" by name
    let init_module = modules
        .iter()
        .find(|m| m.cmdline.contains("init"))
        .or_else(|| modules.iter().find(|m| m.name.contains("sprout")));

    if let Some(mod_desc) = init_module {
        kdebug!(
            "Found init module: {} (cmdline: '{}'), loading...",
            mod_desc.name,
            mod_desc.cmdline
        );

        let aspace = runtime.tasking().make_user_address_space();
        let _hook = GlobalAllocHook;

        // Load Sprout
        let (user_entry, stack_info, mut regions, _aux_info) =
            crate::task::loader::load_module(runtime, aspace, mod_desc)
                .expect("Failed to load sprout");

        // Prepare Module Registry Page
        let reg_phys = crate::memory::alloc_frame().expect("OOM Registry");
        let reg_virt = reg_phys + runtime.phys_to_virt_offset();

        // Fill registry
        unsafe {
            let count = modules.len();
            let ptr = reg_virt as *mut usize;
            *ptr = count; // First word = count

            // Array of ModuleEntry { ptr: usize, len: usize } starts at offset 8 (64-bit)
            // Strings start after array. Max 16 modules typical, but let's calculate.
            // entry_size = 16 bytes.
            let array_start = ptr.add(1) as *mut usize;
            let mut string_offset_bytes = 8 + (count * 16);

            for (i, m) in modules.iter().enumerate() {
                let name_bytes = m.name.as_bytes();
                let name_len = name_bytes.len();

                // Safety check for page overflow
                if string_offset_bytes + name_len > 4096 {
                    kinfo!("Warning: Module registry page overflow, truncating list.");
                    *ptr = i; // Update count
                    break;
                }

                // Copy string
                let string_dst = (reg_virt as *mut u8).add(string_offset_bytes);
                core::ptr::copy_nonoverlapping(name_bytes.as_ptr(), string_dst, name_len);

                // Write Entry (ptr, len)
                let entry_slot = array_start.add(i * 2);
                *entry_slot = 0x600000 + string_offset_bytes; // User virtual address
                *entry_slot.add(1) = name_len;

                string_offset_bytes += name_len;
            }
        }

        // Map Registry to fixed user address 0x600000
        // We map it read-only for user
        runtime
            .tasking()
            .map_page(
                aspace,
                0x600000,
                reg_phys,
                MapPerms {
                    user: true,
                    read: true,
                    write: false,
                    exec: false,
                    kind: MapKind::Normal,
                },
                MapKind::Normal,
                &GlobalAllocHook,
            )
            .unwrap();

        regions.push(VmRegionInfo {
            start: 0x600000,
            end: 0x601000,
            prot: VmProt::USER | VmProt::READ,
            flags: VmMapFlags::empty(),
            backing_kind: VmBackingKind::Unknown,
            _reserved: [0; 7],
        });

        // Flush TLB by reloading CR3
        runtime.tasking().activate_address_space(aspace);

        kdebug!("Spawning sprout with registry at 0x600000...");
        unsafe {
            contract!("Spawning init process...");
            let mut entry = user_entry;
            entry.arg0 = StartupArg::BootRegistry.to_raw(); // arg0 = registry ptr
            // Spawn at Normal priority - all tasks share the same priority for fair scheduling
            crate::sched::spawn_user_task_full::<R>(
                entry,
                aspace,
                stack_info,
                regions,
                crate::task::TaskPriority::Normal,
            );
        }
    } else {
        kdebug!("Sprout not found. Checking fallback...");

        let spawned_fallback = false;
        #[cfg(feature = "diagnostic-apps")]
        {
            if let Some(mod_desc) = modules.iter().find(|m| m.name.contains("threads_demo")) {
                kdebug!("Found threads_demo fallback...");
                let aspace = runtime.tasking().make_user_address_space();
                let (user_entry, stack_info, regions) =
                    crate::task::loader::load_module(runtime, aspace, mod_desc)
                        .expect("Failed to load threads_demo");
                unsafe {
                    crate::sched::spawn_user_task_full::<R>(
                        user_entry,
                        aspace,
                        stack_info,
                        regions,
                        crate::task::TaskPriority::Normal,
                    );
                }
                spawned_fallback = true;
            }
        }

        if !spawned_fallback {
            kdebug!("No modules found. Checking threads_supported...");
            if runtime.threads_supported() {
                kdebug!("Spawning initial threads...");
                kdebug!("Spawning Thread A...");
                crate::task::spawn::<R>(
                    thread_a,
                    StartupArg::Raw(1),
                    crate::task::TaskPriority::Normal,
                    crate::task::Affinity::Any,
                );
                kdebug!("Spawning Thread B...");
                crate::task::spawn::<R>(
                    thread_b,
                    StartupArg::Raw(2),
                    crate::task::TaskPriority::Normal,
                    crate::task::Affinity::Any,
                );
            }
        }
    }

    contract!("Entering scheduler loop.");
    loop {
        crate::task::yield_now::<R>();
        // runtime.wait_for_interrupt(); // TODO: Only call when runqueue is empty
    }
}

extern "C" fn thread_a(arg: usize) -> ! {
    let mut count: usize = 0;
    loop {
        // Only log the first few iterations to avoid flooding serial output
        if count < 5 {
            let ticks = runtime_base().mono_ticks();
            crate::kinfo!("Thread A (arg={}) ticks={}", arg, ticks);
        }
        count = count.wrapping_add(1);
        for _ in 0..1000000 {
            core::hint::black_box(());
        }
        unsafe {
            crate::sched::yield_now_current();
        }
    }
}

extern "C" fn thread_b(arg: usize) -> ! {
    let mut count: usize = 0;
    loop {
        // Only log the first few iterations to avoid flooding serial output
        if count < 5 {
            let ticks = runtime_base().mono_ticks();
            crate::kinfo!("Thread B (arg={}) ticks={}", arg, ticks);
        }
        count = count.wrapping_add(1);
        for _ in 0..1000000 {
            core::hint::black_box(());
        }
        unsafe {
            crate::sched::yield_now_current();
        }
    }
}

pub mod boot_info;

extern "C" fn kernel_secondary_entry<R: BootRuntime>(cpu_index: usize) -> ! {
    // CRITICAL: First, load the kernel's GDT/IDT and set GS_BASE on this secondary CPU
    // This must happen before ANY kernel code that might fault or use logging (which uses GS).
    let base = unsafe { RAW_RUNTIME_BASE.expect("RAW_RUNTIME_BASE not initialized") };
    base.init_secondary_cpu(cpu_index);
    crate::kinfo!(
        "SMP: kernel_secondary_entry arg_cpu={} runtime_cpu={}",
        cpu_index,
        base.current_cpu_index()
    );
    // Verification done via base properties later if needed

    crate::kdebug!("SMP: Entering kernel_secondary_entry for CPU {}", cpu_index);

    // Per-CPU init
    base.mono_ticks(); // ok for logging
    // IMPORTANT: per-CPU SIMD init
    base.simd_init_cpu();

    // Then:
    unsafe {
        crate::sched::cpu_online::<R>(cpu_index);
        crate::sched::enter_secondary(cpu_index);
    }
}

pub fn scan_pci() {
    let rt = runtime_base();
    let mut reg = crate::device_registry::REGISTRY.lock();

    for bus in 0..16 {
        // Bus range restricted for speed in QEMU
        for dev in 0..32 {
            for func in 0..8 {
                let vendor_device = match rt.pci_cfg_read32(bus, dev, func, 0x00) {
                    Ok(val) => val,
                    Err(_) => 0xFFFFFFFF,
                };
                let vendor_id = (vendor_device & 0xFFFF) as u16;
                let device_id = (vendor_device >> 16) as u16;

                if vendor_id == 0xFFFF {
                    if func == 0 {
                        break;
                    } // Next device
                    continue; // Next function
                }

                // Enable Memory Space (bit 1) and Bus Mastering (bit 2)
                let cmd = rt.pci_cfg_read32(bus, dev, func, 0x04).unwrap_or(0);
                let _ = rt.pci_cfg_write32(bus, dev, func, 0x04, cmd | 0x06);

                let class_rev = rt.pci_cfg_read32(bus, dev, func, 0x08).unwrap_or(0);
                let class_code = (class_rev >> 24) as u8;
                let subclass = (class_rev >> 16) as u8;
                let prog_if = (class_rev >> 8) as u8;

                let header_type =
                    (rt.pci_cfg_read32(bus, dev, func, 0x0C).unwrap_or(0) >> 16) as u8;

                let mut bars = [0u64; 6];
                let mut sizes = [0u64; 6];

                // For simplicity, only scan BARs for header type 0 (normal devices)
                if (header_type & 0x7F) == 0 {
                    let mut i = 0;
                    while i < 6 {
                        let offset = 0x10 + (i * 4) as u8;
                        let bar = rt.pci_cfg_read32(bus, dev, func, offset).unwrap_or(0);
                        if bar != 0 {
                            // Check size by writing 0xFFFFFFFF
                            let _ = rt.pci_cfg_write32(bus, dev, func, offset, 0xFFFFFFFF);
                            let size_mask = rt.pci_cfg_read32(bus, dev, func, offset).unwrap_or(0);
                            let _ = rt.pci_cfg_write32(bus, dev, func, offset, bar);

                            if bar & 1 == 0 {
                                // Memory space
                                let is_64 = (bar & 0x4) != 0;
                                let mut final_bar = (bar & 0xFFFFFFF0) as u64;
                                let mut final_size_mask = if is_64 && i < 5 {
                                    let next_offset = offset + 4;
                                    let bar_hi =
                                        rt.pci_cfg_read32(bus, dev, func, next_offset).unwrap_or(0);
                                    let _ =
                                        rt.pci_cfg_write32(bus, dev, func, next_offset, 0xFFFFFFFF);
                                    let size_mask_hi =
                                        rt.pci_cfg_read32(bus, dev, func, next_offset).unwrap_or(0);
                                    let _ = rt.pci_cfg_write32(bus, dev, func, next_offset, bar_hi);

                                    final_bar |= (bar_hi as u64) << 32;
                                    (size_mask & 0xFFFFFFF0) as u64 | ((size_mask_hi as u64) << 32)
                                } else {
                                    (size_mask & 0xFFFFFFF0) as u64 | 0xFFFFFFFF_00000000
                                };

                                let size = (!final_size_mask).wrapping_add(1);
                                bars[i as usize] = final_bar;
                                sizes[i as usize] = size;

                                crate::ktrace!(
                                    "  BAR{} (MEM{}): 0x{:08x} (size 0x{:x})",
                                    i,
                                    if is_64 { "64" } else { "32" },
                                    final_bar,
                                    size
                                );

                                if is_64 {
                                    i += 1; // Skip next slot
                                }
                            } else {
                                // I/O space
                                let size = (!(size_mask & 0xFFFFFFFC)).wrapping_add(1) as u64;
                                bars[i as usize] = (bar & 0xFFFFFFFC) as u64;
                                sizes[i as usize] = size;
                                crate::ktrace!(
                                    "  BAR{} (I/O):  0x{:04x} (size 0x{:x})",
                                    i,
                                    bars[i as usize],
                                    size
                                );
                            }
                        }
                        i += 1;
                    }
                }

                let resource_id =
                    0x2000_0000 | ((bus as u64) << 16) | ((dev as u64) << 8) | (func as u64);

                let entry = crate::device_registry::DeviceEntry {
                    kind: "pci_device",
                    ioport_ranges: &[],
                    resource_id,
                    mmio_bars: bars,
                    mmio_sizes: sizes,
                    vendor_id,
                    device_id,
                    class_code,
                    subclass,
                    prog_if,
                    pci_location: Some(crate::device_registry::PciLocation { bus, dev, func }),
                    msi_cap: None,
                    msix_cap: None,
                    irq_mode: crate::device_registry::IrqMode::Legacy,
                    irq_vector: 0,
                };

                if let Some(idx) = reg.register(entry) {
                    crate::kdebug!(
                        "PCI: Discovered 0x{:04x}:0x{:04x} at {:02x}:{:02x}.{} class={:02x}{:02x}{:02x} id={}",
                        vendor_id,
                        device_id,
                        bus,
                        dev,
                        func,
                        class_code,
                        subclass,
                        prog_if,
                        idx
                    );
                }

                if func == 0 && (header_type & 0x80) == 0 {
                    break;
                }
            }
        }
    }
}
