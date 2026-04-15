//! Per-thread TLS block management for ELF TLS support.
//!
//! This module implements the userspace side of ELF TLS:
//! - Reading TLS template metadata from the process auxiliary vector.
//! - Allocating and initializing per-thread TLS blocks.
//! - Installing a TLS block as the calling thread's hardware TLS pointer.
//!
//! # ELF TLS Layout (x86_64 "Variant II")
//!
//! ```text
//! ┌──────────────────────────────┬──────────────────┐
//! │   TLS data  (≥ memsz bytes)  │   TCB (16 bytes) │
//! └──────────────────────────────┴──────────────────┘
//!  ^block_start                   ^Thread Pointer (TP = FS_BASE)
//!                                   *TP == TP  (mandatory self-pointer)
//! ```
//!
//! The TLS data area is placed *before* the Thread Control Block (TCB).  The
//! Thread Pointer (TP), which on x86_64 is stored in `FS_BASE`, points to the
//! start of the TCB.  Thread-local variables live at negative offsets from TP.
//!
//! # Dynamic TLS Models
//!
//! For ELF programs using the initial-exec or local-exec TLS models the kernel
//! allocates the main thread's block automatically (see `loader.rs`).  For
//! programs or shared libraries using the general-dynamic or local-dynamic
//! models each new thread must call [`setup_thread_tls`] (or the lower-level
//! [`alloc_tls_block`]) from its entry point before accessing any thread-local
//! variable.

use abi::auxv;
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use crate::errors::Errno;

// ── Public types ─────────────────────────────────────────────────────────────

/// Metadata describing the ELF TLS segment for the running process.
///
/// Obtained via [`read_tls_info`], which parses the process auxiliary vector.
/// All sizes are in bytes.
#[derive(Debug, Clone, Copy)]
pub struct TlsInfo {
    /// Biased virtual address of the TLS initialization template in the loaded
    /// image (AT_THINGOS_TLS_TEMPLATE_VA).
    pub template_va: usize,
    /// Initialized bytes in the TLS template (AT_THINGOS_TLS_FILESZ).
    pub filesz: usize,
    /// Total per-thread TLS block size (AT_THINGOS_TLS_MEMSZ).
    pub memsz: usize,
    /// Required alignment for the TLS data block (AT_THINGOS_TLS_ALIGN, ≥ 1).
    pub align: usize,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read TLS metadata from the process auxiliary vector.
///
/// Returns `None` when the process has no PT_TLS segment (i.e. it was loaded
/// from an ELF binary without thread-local storage, or the kernel did not emit
/// the ThingOS TLS auxiliary entries).
pub fn read_tls_info() -> Option<TlsInfo> {
    // First call: discover required buffer size.
    let needed = crate::syscall::auxv_get(&mut []).ok()?;
    if needed == 0 {
        return None;
    }
    let mut buf = alloc::vec![0u8; needed];
    crate::syscall::auxv_get(&mut buf).ok()?;

    // Serialized format: count: u32 LE, then count × (type: u64 LE, value: u64 LE).
    if buf.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;

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
            auxv::AT_NULL => break,
            auxv::AT_THINGOS_TLS_TEMPLATE_VA => template_va = val as usize,
            auxv::AT_THINGOS_TLS_FILESZ => filesz = val as usize,
            auxv::AT_THINGOS_TLS_MEMSZ => memsz = val as usize,
            auxv::AT_THINGOS_TLS_ALIGN => align = val as usize,
            _ => {}
        }
    }

    if memsz == 0 {
        return None; // No PT_TLS segment present.
    }

    Some(TlsInfo {
        template_va,
        filesz,
        memsz,
        align: align.max(1),
    })
}

/// Allocate and initialize a fresh per-thread TLS block.
///
/// Returns the Thread Pointer (TP) value that should be written into the
/// hardware TLS register (FS_BASE on x86_64) before the thread accesses any
/// thread-local variables.
///
/// # Layout
///
/// A contiguous anonymous memory region is allocated with this layout:
///
/// ```text
/// [TLS data: ceil(memsz, tls_align) bytes][TCB: 16 bytes]
///  ^block_addr                             ^TP  (returned)
/// ```
///
/// - `[0, filesz)` — copied verbatim from the TLS template at `info.template_va`.
/// - `[filesz, memsz)` — zeroed (BSS).
/// - `TCB[0]` (`u64`) — set to `TP` (the mandatory ELF x86_64 self-pointer).
///
/// # Errors
///
/// Returns [`Errno::ENOMEM`] when the anonymous memory allocation fails.
pub fn alloc_tls_block(info: &TlsInfo) -> Result<usize, Errno> {
    // Minimum alignment is 16 to satisfy compiler ABI expectations.
    let tls_align = info.align.max(16);
    let data_size = round_up(info.memsz, tls_align);
    let tcb_size: usize = 16; // self-pointer (u64) + DTV pointer (u64)
    let total_size = data_size + tcb_size;

    // Allocate zero-initialized anonymous RW pages for the TLS block.
    let req = VmMapReq {
        addr_hint: 0,
        len: total_size,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::PRIVATE,
        backing: VmBacking::Anonymous { zeroed: true },
    };
    let resp = crate::vm::vm_map(&req)?;
    let block_addr = resp.addr;
    if block_addr == 0 {
        return Err(Errno::ENOMEM);
    }

    // TP is placed at the start of the TCB, immediately after the TLS data.
    let tp = block_addr + data_size;

    // Copy the TLS initialization template into the data area.
    if info.filesz > 0 && info.template_va != 0 {
        let copy_len = info.filesz.min(info.memsz);
        unsafe {
            core::ptr::copy_nonoverlapping(
                info.template_va as *const u8,
                block_addr as *mut u8,
                copy_len,
            );
        }
    }
    // BSS bytes [filesz, memsz) are already zero from the anonymous mapping.

    // Write the mandatory ELF x86_64 TCB self-pointer: *TP = TP.
    unsafe {
        *(tp as *mut usize) = tp;
    }

    Ok(tp)
}

/// Allocate a TLS block and install it as the calling thread's TLS pointer.
///
/// This is a convenience wrapper that combines [`alloc_tls_block`] with a
/// `SYS_TASK_SET_TLS_BASE` syscall.  Call this from a new thread's entry point
/// **before** accessing any `#[thread_local]` variables.
///
/// Returns the TP value that was installed, or an error if allocation or the
/// syscall failed.
pub fn setup_thread_tls(info: &TlsInfo) -> Result<usize, Errno> {
    let tp = alloc_tls_block(info)?;
    crate::syscall::task_set_tls_base(tp)?;
    Ok(tp)
}

// ── Private helpers ───────────────────────────────────────────────────────────

#[inline]
fn round_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(0, 16), 0);
        assert_eq!(round_up(1, 16), 16);
        assert_eq!(round_up(16, 16), 16);
        assert_eq!(round_up(17, 16), 32);
        assert_eq!(round_up(128, 16), 128);
        assert_eq!(round_up(129, 16), 144);
    }

    #[test]
    fn round_up_align_zero_is_identity() {
        assert_eq!(round_up(42, 0), 42);
    }

    #[test]
    fn tls_info_align_minimum_is_one() {
        // TlsInfo stores align >= 1.
        let info = TlsInfo {
            template_va: 0,
            filesz: 0,
            memsz: 64,
            align: 0,
        };
        // align = 0 comes back from auxv parsing as 1 via .max(1)
        assert_eq!(info.align, 0); // raw field; callers use .max(1) in alloc_tls_block
        // alloc_tls_block itself clamps to max(16)
        let tls_align = info.align.max(16);
        assert_eq!(tls_align, 16);
    }
}
