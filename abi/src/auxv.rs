//! ELF Auxiliary Vector (`AT_*`) type constants.
//!
//! The auxiliary vector is a list of `(type, value)` pairs passed from the
//! kernel to userspace in `SYS_AUXV_GET`.  These constants identify the type
//! field of each entry.
//!
//! Standard entries follow the Linux/SysV ABI numbering so that generic ELF
//! loaders and runtimes can interoperate.  Janix-specific extensions occupy
//! the `0x1000–0x100f` range.

// ── Standard AT_* constants (Linux/SysV ABI) ─────────────────────────────────

/// End of the auxiliary vector.
pub const AT_NULL: u64 = 0;
/// Entry ignored.
pub const AT_IGNORE: u64 = 1;
/// File descriptor of the program.
pub const AT_EXECFD: u64 = 2;
/// Program headers for the program.
pub const AT_PHDR: u64 = 3;
/// Size of one program header entry.
pub const AT_PHENT: u64 = 4;
/// Number of program header entries.
pub const AT_PHNUM: u64 = 5;
/// System page size.
pub const AT_PAGESZ: u64 = 6;
/// Base address of the interpreter (dynamic linker).
pub const AT_BASE: u64 = 7;
/// Flags for the program.
pub const AT_FLAGS: u64 = 8;
/// Program entry point.
pub const AT_ENTRY: u64 = 9;

// ── Janix-specific AT_* extensions (0x1000–0x100f) ───────────────────────────

/// Biased virtual address of the ELF TLS initialization template in the
/// loaded image (PT_TLS `p_vaddr` + load bias).
///
/// The TLS template contains the initial values for thread-local variables.
/// A dynamic linker or runtime copies this region when allocating a new
/// per-thread TLS block.  Zero when the executable has no PT_TLS segment.
pub const AT_JANIX_TLS_TEMPLATE_VA: u64 = 0x1000;

/// Size of the initialized portion of the TLS template (PT_TLS `p_filesz`).
///
/// Only the first `AT_JANIX_TLS_FILESZ` bytes of the template region contain
/// explicit values.  The remainder up to `AT_JANIX_TLS_MEMSZ` must be
/// zero-initialized (BSS).
pub const AT_JANIX_TLS_FILESZ: u64 = 0x1001;

/// Total per-thread TLS block size in bytes (PT_TLS `p_memsz`).
///
/// When creating a new thread, allocate at least this many bytes (aligned to
/// `AT_JANIX_TLS_ALIGN`) for the TLS data area, placed *before* the Thread
/// Control Block (TCB) in the ELF x86_64 "variant II" layout.
pub const AT_JANIX_TLS_MEMSZ: u64 = 0x1002;

/// Required alignment for the per-thread TLS data block (PT_TLS `p_align`).
pub const AT_JANIX_TLS_ALIGN: u64 = 0x1003;
