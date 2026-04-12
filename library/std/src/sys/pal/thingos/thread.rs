//! ThingOS PAL — `std::thread` support.
//!
//! # Architecture
//!
//! Thread creation delegates to the ThingOS kernel via `SYS_SPAWN_THREAD`
//! (0x1004), which accepts a pointer to a `SpawnThreadReq` structure describing
//! the entry point, stack layout, argument, and initial TLS base.
//!
//! # TLS model — Option A (native ELF TLS)
//!
//! ThingOS uses native ELF TLS (Variant II on x86_64): the thread pointer
//! (`FS_BASE`) points to the Thread Control Block (TCB), and thread-local
//! variables live at negative offsets relative to the TCB.  Each spawned
//! thread receives its own TLS block, pre-populated by copying the TLS
//! template read from the kernel auxiliary vector.
//!
//! `"has-thread-local": true` in every ThingOS target JSON causes rustc to set
//! the `target_thread_local` cfg, which routes `thread_local!` to the native
//! (`#[thread_local]`) implementation instead of the library-based OS-key path.
//!
//! # TLS destructors
//!
//! TLS destructors are **not** run by the OS on thread exit.  Instead the
//! thread trampoline calls `sys::thread_local::destructors::run()` manually
//! at the end of each spawned thread (same pattern as the Hermit port).
//! `guard::enable()` is a no-op for ThingOS because there is no kernel-level
//! thread-exit callback.
//!
//! # Stack layout
//!
//! ```text
//! low addr ───────────────────────────────────── high addr
//! [ guard page (4 KiB, unmapped) | reserve | committed ]
//!                                            ^sp (reserve_end − 8)
//! ```
//!
//! The kernel validates the `StackInfo` fields before accepting the spawn
//! request; see `abi/src/types/system.rs` for the exact constraints.
//!
//! # Panic behaviour
//!
//! ThingOS uses `panic = "abort"` in all current targets, so a panicking
//! thread aborts the whole process immediately via the `abort_internal` hook.
//! `join()` therefore never returns `Err(...)` from a panic — if the child
//! panics the process is already gone.  This is documented as a current
//! limitation; a future port with `panic = "unwind"` should capture the panic
//! payload and propagate it through the join handle in the usual way.

#[cfg(panic = "unwind")]
compile_error!(
    "ThingOS std thread PAL is abort-only: panic=unwind is not supported for target_os=thingos"
);

use crate::ffi::CStr;
use crate::num::NonZero;
use crate::thread::ThreadInit;
use crate::time::Duration;

// ── Syscall numbers (abi/src/numbers.rs) ─────────────────────────────────────

const SYS_EXIT: u32 = 0x1000;
const SYS_GET_TID: u32 = 0x1001;
const SYS_SPAWN_THREAD: u32 = 0x1004;
const SYS_TASK_WAIT: u32 = 0x1007;
const SYS_YIELD: u32 = 0x100B;
const SYS_SLEEP_NS: u32 = 0x1200;
const SYS_AUXV_GET: u32 = 0x1105;
const SYS_VM_MAP: u32 = 0x2001;

// ── Local raw_syscall6 wrapper ────────────────────────────────────────────────

#[inline(always)]
unsafe fn raw_syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    unsafe { crate::sys::pal::raw_syscall6(n, a0, a1, a2, a3, a4, a5) }
}

// ── ABI struct mirrors ────────────────────────────────────────────────────────
//
// These must match the kernel layout exactly (abi/src/types/system.rs and
// abi/src/vm.rs).  They are defined inline here to avoid a hard dependency on
// the abi crate from within std.

/// Mirrors `abi::types::StackInfo` (6 × usize = 48 bytes on 64-bit).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct StackInfo {
    guard_start: usize,
    guard_end: usize,
    reserve_start: usize,
    reserve_end: usize,
    committed_start: usize,
    grow_chunk_bytes: usize,
}

/// Mirrors `abi::types::SpawnThreadReq` (88 bytes on 64-bit).
#[repr(C)]
struct SpawnThreadReq {
    entry: usize,
    sp: usize,
    arg: usize,
    stack: StackInfo,
    tls_base: usize,
    flags: u32,
    _pad: u32,
}

/// Anonymous-only variant of `abi::vm::VmMapReq` (48 bytes on 64-bit).
///
/// `VmBacking::Anonymous { zeroed: bool }` has the C layout:
///   - tag (u32) at offset 0
///   - padding (u32) at offset 4
///   - zeroed (u8) at offset 8
///   - padding ([u8; 15]) at offset 9
/// giving a total `backing` size of 24 bytes and `VmMapReq` size of 48 bytes.
#[repr(C)]
struct VmMapReqAnon {
    addr_hint: usize,
    len: usize,
    prot: u32,
    flags: u32,
    // VmBacking::Anonymous { zeroed }
    backing_tag: u32,            // discriminant = 0 (Anonymous)
    _backing_pad: u32,           // padding before union body
    backing_zeroed: u8,          // zeroed = true (1) or false (0)
    _backing_body_pad: [u8; 15], // pad union body to 16 bytes (size of File variant)
}

/// Mirrors `abi::vm::VmMapResp` (16 bytes on 64-bit).
#[repr(C)]
#[derive(Default)]
struct VmMapResp {
    addr: usize,
    len: usize,
}

// VmProt bit flags (abi/src/vm.rs)
const PROT_READ: u32 = 1 << 0;
const PROT_WRITE: u32 = 1 << 1;
const PROT_USER: u32 = 1 << 3;

// VmMapFlags bit flags (abi/src/vm.rs)
const MAP_FIXED: u32 = 1 << 0;
const MAP_GUARD: u32 = 1 << 1;
const MAP_PRIVATE: u32 = 1 << 2;

// ── Stack parameters ──────────────────────────────────────────────────────────

const PAGE_SIZE: usize = 4096;
const DEFAULT_RESERVE: usize = 2 * 1024 * 1024; // 2 MiB virtual reserve
const INITIAL_COMMIT: usize = 64 * 1024; // 64 KiB initially committed
const GROW_CHUNK: usize = 64 * 1024; // 64 KiB per growth step

pub const DEFAULT_MIN_STACK_SIZE: usize = DEFAULT_RESERVE;

// ── Thread handle ─────────────────────────────────────────────────────────────

/// A spawned ThingOS thread, identified by its kernel TID.
pub struct Thread {
    tid: u64,
}

unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    /// Spawn a new thread.
    ///
    /// # Safety
    /// See `thread::Builder::spawn_unchecked`.
    pub unsafe fn new(stack_size: usize, init: Box<ThreadInit>) -> crate::io::Result<Thread> {
        // Leak the init box; it will be reconstructed and freed in thread_start.
        let data = Box::into_raw(init);

        // Allocate the thread stack.
        let reserve = stack_size.max(DEFAULT_RESERVE);
        let (sp, stack_info) = match unsafe { allocate_stack(reserve) } {
            Ok(v) => v,
            Err(e) => {
                // Reclaim the init box on failure to avoid a leak.
                drop(unsafe { Box::from_raw(data) });
                return Err(e);
            }
        };

        // Allocate a per-thread TLS block populated from the process TLS template.
        // Returns 0 when the process has no PT_TLS segment (no TLS variables); the
        // kernel then leaves FS_BASE in its default initial state (0 on x86_64).
        let tls_base = allocate_tls_block();

        let req = SpawnThreadReq {
            entry: thread_start as *const () as usize,
            sp,
            arg: data as usize,
            stack: stack_info,
            tls_base,
            flags: 0,
            _pad: 0,
        };

        let ret = unsafe {
            raw_syscall6(SYS_SPAWN_THREAD, &req as *const SpawnThreadReq as usize, 0, 0, 0, 0, 0)
        };

        if ret < 0 {
            drop(unsafe { Box::from_raw(data) });
            return Err(crate::io::Error::from_raw_os_error((-ret) as i32));
        }

        Ok(Thread { tid: ret as u64 })
    }

    /// Block until the thread exits.
    pub fn join(self) {
        unsafe {
            raw_syscall6(SYS_TASK_WAIT, self.tid as usize, 0, 0, 0, 0, 0);
        }
    }
}

// ── Thread entry trampoline ───────────────────────────────────────────────────

/// Entry point for every spawned std thread.
///
/// By the time this function runs, the kernel has already loaded `tls_base`
/// into the hardware TLS register (FS_BASE on x86_64), so `#[thread_local]`
/// variables are immediately accessible.
#[inline(never)]
extern "C" fn thread_start(data: usize) -> ! {
    // Reconstruct the ThreadInit box that was leaked in Thread::new.
    // SAFETY: `data` is the pointer returned by Box::into_raw in Thread::new.
    let init =
        unsafe { Box::from_raw(crate::ptr::with_exposed_provenance_mut::<ThreadInit>(data)) };

    // Initialize the thread-current handle and retrieve the Rust entry closure.
    let rust_start = init.init();

    // Execute the user closure.
    rust_start();

    // Run any TLS destructors that were registered during the thread's lifetime.
    // With `target_thread_local` the destructor list is stored in a per-thread
    // #[thread_local] static so this is always safe to call.
    unsafe {
        crate::sys::thread_local::destructors::run();
    }

    // Notify the runtime that this thread is about to exit.
    crate::rt::thread_cleanup();

    // Exit this thread.  SYS_EXIT with code 0; this call does not return.
    unsafe {
        raw_syscall6(SYS_EXIT, 0, 0, 0, 0, 0, 0);
        core::hint::unreachable_unchecked()
    }
}

// ── Yield / sleep / misc ──────────────────────────────────────────────────────

/// Yield the current thread's time-slice back to the scheduler.
///
/// Backed by `SYS_YIELD` (0x100B).  Calling this from a busy-loop allows
/// peer threads to make progress on the same CPU.
pub fn yield_now() {
    unsafe {
        raw_syscall6(SYS_YIELD, 0, 0, 0, 0, 0, 0);
    }
}

/// Sleep for the specified duration.
///
/// Backed by `SYS_SLEEP_NS` (0x1200), which takes nanoseconds as a `usize`.
/// Durations longer than `usize::MAX` nanoseconds are clamped.
pub fn sleep(dur: Duration) {
    let ns = dur.as_nanos();
    let ns_usize = if ns > usize::MAX as u128 { usize::MAX } else { ns as usize };
    unsafe {
        raw_syscall6(SYS_SLEEP_NS, ns_usize, 0, 0, 0, 0, 0);
    }
}

/// Return the number of hardware threads available.
///
/// Currently returns 1 until the scheduler exposes a CPU-count syscall.
pub fn available_parallelism() -> crate::io::Result<NonZero<usize>> {
    // SAFETY: 1 is non-zero.
    Ok(unsafe { NonZero::new_unchecked(1) })
}

/// Return the OS-level thread identifier of the calling thread.
pub fn current_os_id() -> Option<u64> {
    let ret = unsafe { raw_syscall6(SYS_GET_TID, 0, 0, 0, 0, 0, 0) };
    if ret < 0 { None } else { Some(ret as u64) }
}

/// Set the OS-level thread name.  Not yet implemented on ThingOS.
pub fn set_name(_name: &CStr) {
    // Thread naming not yet supported by the kernel.
}

// ── Stack allocation ──────────────────────────────────────────────────────────

/// Allocate a thread stack and return `(sp, StackInfo)`.
///
/// Layout:
/// ```text
/// [ guard (PAGE_SIZE, unmapped) ][ reserve ][ committed ]
///  ^base                          ^guard_end  ^commit_start  ^sp = reserve_end-8
/// ```
///
/// The guard page is a virtual reservation without physical backing; any
/// access to it causes a page fault, providing overflow detection.
unsafe fn allocate_stack(reserve_bytes: usize) -> crate::io::Result<(usize, StackInfo)> {
    let guard_bytes = PAGE_SIZE;
    let reserve = align_up(reserve_bytes, PAGE_SIZE);
    let commit = align_up(INITIAL_COMMIT, PAGE_SIZE);
    let total = guard_bytes + reserve;

    // Phase 1: reserve the full virtual range as a guard (no physical pages).
    let mut guard_req = VmMapReqAnon {
        addr_hint: 0,
        len: total,
        prot: PROT_USER, // user-accessible but no R/W → page fault on access
        flags: MAP_GUARD | MAP_PRIVATE,
        backing_tag: 0,
        _backing_pad: 0,
        backing_zeroed: 1,
        _backing_body_pad: [0; 15],
    };
    let mut resp = VmMapResp::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_VM_MAP,
            &mut guard_req as *mut VmMapReqAnon as usize,
            &mut resp as *mut VmMapResp as usize,
            0,
            0,
            0,
            0,
        )
    };
    if ret < 0 {
        return Err(crate::io::Error::from_raw_os_error((-ret) as i32));
    }
    let base = resp.addr;

    // Compute region boundaries.
    let guard_start = base;
    let guard_end = base + guard_bytes;
    let reserve_start = guard_end;
    let reserve_end = base + total;
    let commit_start = reserve_end - commit;

    // Phase 2: commit the top portion of the reserve with R+W permissions.
    let mut commit_req = VmMapReqAnon {
        addr_hint: commit_start,
        len: commit,
        prot: PROT_USER | PROT_READ | PROT_WRITE,
        flags: MAP_FIXED | MAP_PRIVATE,
        backing_tag: 0,
        _backing_pad: 0,
        backing_zeroed: 1,
        _backing_body_pad: [0; 15],
    };
    let mut commit_resp = VmMapResp::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_VM_MAP,
            &mut commit_req as *mut VmMapReqAnon as usize,
            &mut commit_resp as *mut VmMapResp as usize,
            0,
            0,
            0,
            0,
        )
    };
    if ret < 0 {
        return Err(crate::io::Error::from_raw_os_error((-ret) as i32));
    }

    // SP points to the top of the reserve (8 bytes below the top for alignment).
    let sp = reserve_end - 8;

    let stack_info = StackInfo {
        guard_start,
        guard_end,
        reserve_start,
        reserve_end,
        committed_start: commit_start,
        grow_chunk_bytes: GROW_CHUNK,
    };

    Ok((sp, stack_info))
}

// ── TLS block allocation ──────────────────────────────────────────────────────

/// Allocate a per-thread TLS block from the process TLS template.
///
/// Returns the Thread Pointer (TP = FS_BASE value) on success, or `0` when:
/// - the process has no `PT_TLS` segment (no thread-local variables), or
/// - memory allocation fails.
///
/// The block layout follows the ELF Variant II model used on x86_64:
/// ```text
/// [ TLS data area (memsz bytes, negative offsets from TP) ][ TCB (16 bytes) ]
///  ^block_addr                                              ^TP = thread pointer
/// ```
/// The mandatory ELF Variant II self-pointer `*TP = TP` is written before
/// passing `TP` to `SYS_SPAWN_THREAD`.
fn allocate_tls_block() -> usize {
    let info = match read_tls_info() {
        Some(i) if i.memsz > 0 => i,
        _ => return 0,
    };

    // At least 16-byte alignment to satisfy compiler ABI expectations.
    let tls_align = info.align.max(16);
    let data_size = align_up(info.memsz, tls_align);
    let tcb_size = 16usize; // Variant II TCB: self-pointer (u64) + DTV slot (u64)
    let total = data_size + tcb_size;

    // Allocate zeroed R+W anonymous memory for the TLS block.
    let mut req = VmMapReqAnon {
        addr_hint: 0,
        len: total,
        prot: PROT_USER | PROT_READ | PROT_WRITE,
        flags: MAP_PRIVATE,
        backing_tag: 0,
        _backing_pad: 0,
        backing_zeroed: 1,
        _backing_body_pad: [0; 15],
    };
    let mut resp = VmMapResp::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_VM_MAP,
            &mut req as *mut VmMapReqAnon as usize,
            &mut resp as *mut VmMapResp as usize,
            0,
            0,
            0,
            0,
        )
    };
    if ret < 0 || resp.addr == 0 {
        return 0;
    }
    let block_addr = resp.addr;

    // Thread Pointer = start of TCB, immediately after the TLS data area.
    let tp = block_addr + data_size;

    // Copy the TLS initialization image (only filesz bytes; remainder stays 0).
    if info.filesz > 0 && info.template_va != 0 {
        let copy_len = info.filesz.min(info.memsz);
        unsafe {
            core::ptr::copy_nonoverlapping(
                crate::ptr::with_exposed_provenance::<u8>(info.template_va),
                crate::ptr::with_exposed_provenance_mut::<u8>(block_addr),
                copy_len,
            );
        }
    }

    // Write the ELF Variant II self-pointer: the first word of the TCB must
    // contain the TP itself so that `mov rax, fs:0` returns the TP value.
    unsafe {
        core::ptr::write(crate::ptr::with_exposed_provenance_mut::<usize>(tp), tp);
    }

    tp
}

// ── TLS auxiliary vector reader ───────────────────────────────────────────────

struct TlsInfo {
    template_va: usize, // AT_JANIX_TLS_TEMPLATE_VA
    filesz: usize,      // AT_JANIX_TLS_FILESZ
    memsz: usize,       // AT_JANIX_TLS_MEMSZ
    align: usize,       // AT_JANIX_TLS_ALIGN
}

/// Read TLS metadata from the kernel auxiliary vector via `SYS_AUXV_GET`.
///
/// Format: `count: u32 LE`, then `count` entries of `(type: u64 LE, value: u64 LE)`.
/// The final entry is always `AT_NULL (0, 0)`.
///
/// Returns `None` when:
/// - the auxv syscall fails, or
/// - no `AT_JANIX_TLS_MEMSZ` entry is present (no PT_TLS segment).
fn read_tls_info() -> Option<TlsInfo> {
    const AT_NULL: u64 = 0;
    const AT_JANIX_TLS_TEMPLATE_VA: u64 = 0x1000;
    const AT_JANIX_TLS_FILESZ: u64 = 0x1001;
    const AT_JANIX_TLS_MEMSZ: u64 = 0x1002;
    const AT_JANIX_TLS_ALIGN: u64 = 0x1003;

    // Phase 1: query required buffer size (buf_ptr = 0 → return total bytes).
    let needed = unsafe { raw_syscall6(SYS_AUXV_GET, 0, 0, 0, 0, 0, 0) };
    if needed <= 0 {
        return None;
    }
    let needed = needed as usize;

    // Phase 2: fill the buffer.
    let mut buf = crate::vec![0u8; needed];
    let ret =
        unsafe { raw_syscall6(SYS_AUXV_GET, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0, 0) };
    if ret < 0 || buf.len() < 4 {
        return None;
    }

    let count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;

    let mut template_va = 0usize;
    let mut filesz = 0usize;
    let mut memsz = 0usize;
    let mut align = 0usize;

    for i in 0..count {
        let off = 4 + i * 16;
        if off + 16 > buf.len() {
            break;
        }
        let typ = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
        let val = u64::from_le_bytes(buf[off + 8..off + 16].try_into().ok()?);
        match typ {
            AT_NULL => break,
            AT_JANIX_TLS_TEMPLATE_VA => template_va = val as usize,
            AT_JANIX_TLS_FILESZ => filesz = val as usize,
            AT_JANIX_TLS_MEMSZ => memsz = val as usize,
            AT_JANIX_TLS_ALIGN => align = val as usize,
            _ => {}
        }
    }

    if memsz == 0 {
        return None;
    }

    Some(TlsInfo { template_va, filesz, memsz, align: align.max(1) })
}

// ── Utility ───────────────────────────────────────────────────────────────────

#[inline]
fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "align must be a power of two");
    (value + align - 1) & !(align - 1)
}
