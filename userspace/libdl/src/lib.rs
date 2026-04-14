//! `libdl` — Runtime dynamic loading API for Janix userspace.
//!
//! Provides `dlopen`, `dlsym`, `dlclose`, and `dlerror` analogous to the
//! POSIX `<dlfcn.h>` interface.
//!
//! ## How it works
//!
//! - `dlopen(path, flags)` reads the ELF file from the VFS, maps its
//!   `PT_LOAD` segments into the current address space via `SYS_VM_MAP`,
//!   processes relocations, and returns an opaque handle.
//! - `dlsym(handle, name)` walks the loaded object's ELF dynamic symbol
//!   table to locate a named symbol and returns its absolute address.
//! - `dlclose(handle)` marks the handle slot as free.  Address-space
//!   cleanup (unmapping) is left for a future GC pass; for now the memory
//!   remains mapped but the handle is reusable.
//! - `dlerror()` returns a pointer to a static NUL-terminated ASCII error
//!   string describing the most recent failure, or `null` if there was no
//!   error since the last call.
//!
//! ## Design constraints
//!
//!   heap allocation.
//! - No dependency on the kernel-level `ld.so` interpreter; this library
//!   implements its own ELF parsing and loading so it can be linked
//!   statically into any userspace binary.
//! - Kernel role: none beyond `mmap`/file access (as per the issue spec).
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use alloc::vec::Vec;
use abi::vm::{VmBacking, VmMapFlags, VmProt};
use spin::Mutex;
use stem::syscall::{vfs_close, vfs_open, vfs_read, vfs_seek, vfs_stat};
use stem::vm::{VmMapReq, vm_map};

// ── Public constants ──────────────────────────────────────────────────────────

/// `dlopen` pseudo-handle: search the default (global) symbol namespace.
pub const RTLD_DEFAULT: *mut core::ffi::c_void = core::ptr::null_mut();

/// `dlopen` pseudo-handle: search libraries loaded after the caller.
/// Currently treated the same as `RTLD_DEFAULT`.
pub const RTLD_NEXT: *mut core::ffi::c_void = usize::MAX as *mut core::ffi::c_void;

/// Resolve symbols lazily as they are called (placeholder; we always bind eagerly).
pub const RTLD_LAZY: i32 = 0x001;
/// Resolve all symbols at `dlopen` time.
pub const RTLD_NOW: i32 = 0x002;
/// Make symbols from this library globally visible.
pub const RTLD_GLOBAL: i32 = 0x100;
/// Keep symbols local to this library (default).
pub const RTLD_LOCAL: i32 = 0x000;

// ── ELF constants ─────────────────────────────────────────────────────────────

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;

const DT_NULL: i64 = 0;
#[allow(dead_code)]
const DT_NEEDED: i64 = 1;
#[allow(dead_code)]
const DT_PLTRELSZ: i64 = 2;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_JMPREL: i64 = 23;

const R_X86_64_NONE: u32 = 0;
const R_X86_64_64: u32 = 1;
const R_X86_64_COPY: u32 = 5;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;

const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;

const PAGE_SIZE: usize = 4096;

// ── Handle table ──────────────────────────────────────────────────────────────

/// Maximum number of concurrently open `dlopen` handles.
const MAX_HANDLES: usize = 64;

/// Sentinel index value for a free slot.
const SLOT_FREE: u8 = 0;
/// Sentinel for an in-use slot.
const SLOT_USED: u8 = 1;

/// Information about a single loaded shared object.
#[derive(Copy, Clone)]
pub struct HandleEntry {
    /// Whether this slot is occupied.
    state: u8,
    /// Load bias (absolute VA = file VA + bias).
    pub bias: usize,
    /// Virtual address of `DT_SYMTAB` (already biased).
    pub symtab_va: usize,
    /// Virtual address of `DT_STRTAB` (already biased).
    pub strtab_va: usize,
    /// Value of `DT_STRSZ`.
    pub strsz: usize,
    /// Size of one `Elf64_Sym` entry (from `DT_SYMENT`, usually 24).
    pub syment: usize,
    /// Base address (lowest mapped page).
    pub base: usize,
    /// Total mapped size in bytes (rounded up to page boundary).
    pub map_size: usize,
}

impl HandleEntry {
    const fn empty() -> Self {
        HandleEntry {
            state: SLOT_FREE,
            bias: 0,
            symtab_va: 0,
            strtab_va: 0,
            strsz: 0,
            syment: 24,
            base: 0,
            map_size: 0,
        }
    }
}

struct HandleTable {
    slots: [HandleEntry; MAX_HANDLES],
    /// Rolling base address for placing new libraries.
    next_base: usize,
}

impl HandleTable {
    const fn new() -> Self {
        const E: HandleEntry = HandleEntry::empty();
        HandleTable {
            slots: [E; MAX_HANDLES],
            next_base: 0x5000_0000,
        }
    }

    /// Allocate the next available slot, returning its 1-based index or 0 on
    /// overflow.
    fn alloc(&mut self) -> Option<usize> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.state == SLOT_FREE {
                slot.state = SLOT_USED;
                return Some(i + 1); // 1-based so 0 is "no handle"
            }
        }
        None
    }

    fn free(&mut self, idx: usize) {
        if idx == 0 { return; }
        let i = idx - 1;
        if i < MAX_HANDLES {
            self.slots[i] = HandleEntry::empty();
        }
    }

    fn next_lib_base(&mut self, map_size: usize) -> usize {
        let base = self.next_base;
        self.next_base = (self.next_base + map_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        // Pad by one extra page between libraries to avoid accidental overlap.
        self.next_base += PAGE_SIZE;
        base
    }
}

static HANDLES: Mutex<HandleTable> = Mutex::new(HandleTable::new());

// ── Error state ───────────────────────────────────────────────────────────────

/// Maximum length of the error string including the trailing NUL.
const ERR_BUF_LEN: usize = 128;

struct ErrState {
    buf: [u8; ERR_BUF_LEN],
    /// `true` if an error is pending (dlerror() will return the buffer and
    /// then clear the flag).
    pending: bool,
}

impl ErrState {
    const fn new() -> Self {
        ErrState { buf: [0u8; ERR_BUF_LEN], pending: false }
    }

    fn set(&mut self, msg: &[u8]) {
        let n = msg.len().min(ERR_BUF_LEN - 1);
        self.buf[..n].copy_from_slice(&msg[..n]);
        self.buf[n] = 0;
        self.pending = true;
    }

    fn clear(&mut self) {
        self.pending = false;
    }
}

static ERR_STATE: Mutex<ErrState> = Mutex::new(ErrState::new());

/// Static buffer for the value returned by `dlerror()`.
///
/// We copy from `ErrState` into this buffer while holding the lock, then
/// return a pointer to it.  The pointer remains valid until the next
/// `dlerror()` call.
static mut DLERROR_BUF: [u8; ERR_BUF_LEN] = [0u8; ERR_BUF_LEN];

// ── Helpers ───────────────────────────────────────────────────────────────────

fn set_error(msg: &[u8]) {
    ERR_STATE.lock().set(msg);
}

fn clear_error() {
    ERR_STATE.lock().clear();
}

fn read_u16_le(buf: &[u8], off: usize) -> Option<u16> {
    let s = buf.get(off..off + 2)?;
    Some(u16::from_le_bytes([s[0], s[1]]))
}

fn read_u32_le(buf: &[u8], off: usize) -> Option<u32> {
    let s = buf.get(off..off + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_u64_le(buf: &[u8], off: usize) -> Option<u64> {
    let s = buf.get(off..off + 8)?;
    Some(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
}

unsafe fn read_u16_ptr(p: *const u8, off: usize) -> u16 {
    u16::from_le_bytes([*p.add(off), *p.add(off + 1)])
}

unsafe fn read_u32_ptr(p: *const u8, off: usize) -> u32 {
    u32::from_le_bytes([
        *p.add(off),
        *p.add(off + 1),
        *p.add(off + 2),
        *p.add(off + 3),
    ])
}

unsafe fn read_u64_ptr(p: *const u8, off: usize) -> u64 {
    u64::from_le_bytes([
        *p.add(off),
        *p.add(off + 1),
        *p.add(off + 2),
        *p.add(off + 3),
        *p.add(off + 4),
        *p.add(off + 5),
        *p.add(off + 6),
        *p.add(off + 7),
    ])
}

fn align_down(v: usize, a: usize) -> usize {
    v & !(a - 1)
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// Read a NUL-terminated byte string from memory at `va + offset`, up to
/// `max_len` bytes.
unsafe fn strtab_str(strtab_va: usize, offset: usize, max_len: usize) -> &'static [u8] {
    let ptr = (strtab_va + offset) as *const u8;
    let mut len = 0;
    while len < max_len {
        if *ptr.add(len) == 0 {
            break;
        }
        len += 1;
    }
    core::slice::from_raw_parts(ptr, len)
}

// ── VFS file I/O ──────────────────────────────────────────────────────────────

fn pread(fd: u32, buf: &mut [u8], offset: u64) -> usize {
    let _ = vfs_seek(fd, offset as i64, 0);
    match vfs_read(fd, buf) {
        Ok(n) => n,
        Err(_) => 0,
    }
}

fn read_file(path: &str) -> Option<Vec<u8>> {
    let fd = vfs_open(path, 0).ok()?;
    let stat = vfs_stat(fd).ok()?;
    let size = stat.size as usize;
    if size == 0 || size > 64 * 1024 * 1024 {
        let _ = vfs_close(fd);
        return None;
    }
    let mut buf = alloc::vec![0u8; size];
    let mut pos = 0;
    while pos < size {
        let n = pread(fd, &mut buf[pos..], pos as u64);
        if n == 0 {
            break;
        }
        pos += n;
    }
    let _ = vfs_close(fd);
    if pos < size { None } else { Some(buf) }
}

// ── ELF loading ───────────────────────────────────────────────────────────────

/// Compute the total virtual address span of all `PT_LOAD` segments.
fn elf_load_size(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 64 {
        return None;
    }
    let e_phoff = read_u64_le(bytes, 32)? as usize;
    let e_phentsize = read_u16_le(bytes, 54)? as usize;
    let e_phnum = read_u16_le(bytes, 56)? as usize;
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if read_u32_le(bytes, off)? == PT_LOAD {
            let vaddr = read_u64_le(bytes, off + 16)? as usize;
            let memsz = read_u64_le(bytes, off + 40)? as usize;
            if vaddr < lo {
                lo = vaddr;
            }
            if vaddr + memsz > hi {
                hi = vaddr + memsz;
            }
        }
    }
    if hi > lo { Some(hi - lo) } else { None }
}

/// Map all `PT_LOAD` segments of `elf_bytes` into the address space at
/// `load_base`.  Returns `(bias, symtab_va, strtab_va, strsz, syment, base,
/// map_size)` on success, or an error message on failure.
fn map_elf(
    elf_bytes: &[u8],
    load_base: usize,
) -> Result<(usize, usize, usize, usize, usize, usize, usize), &'static [u8]> {
    if elf_bytes.len() < 64 {
        return Err(b"ELF too small");
    }
    if &elf_bytes[0..4] != b"\x7fELF" {
        return Err(b"bad ELF magic");
    }
    if elf_bytes[4] != 2 || elf_bytes[5] != 1 {
        return Err(b"not ELF64 LE");
    }

    let e_type = read_u16_le(elf_bytes, 16).ok_or(b"truncated ELF header" as &[u8])?;
    let e_phoff =
        read_u64_le(elf_bytes, 32).ok_or(b"truncated ELF header" as &[u8])? as usize;
    let e_phentsize =
        read_u16_le(elf_bytes, 54).ok_or(b"truncated ELF header" as &[u8])? as usize;
    let e_phnum =
        read_u16_le(elf_bytes, 56).ok_or(b"truncated ELF header" as &[u8])? as usize;
    if e_phentsize == 0 || e_phnum == 0 {
        return Err(b"no program headers");
    }

    // Determine minimum vaddr for bias calculation.
    let mut min_vaddr = usize::MAX;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if read_u32_le(elf_bytes, off).unwrap_or(0) == PT_LOAD {
            let vaddr = read_u64_le(elf_bytes, off + 16).unwrap_or(0) as usize;
            if vaddr < min_vaddr {
                min_vaddr = vaddr;
            }
        }
    }
    if min_vaddr == usize::MAX {
        return Err(b"no PT_LOAD segments");
    }

    let bias: usize = if e_type == 3 /* ET_DYN */ {
        load_base.wrapping_sub(min_vaddr)
    } else {
        0
    };

    let base = load_base;
    let mut map_end = load_base;

    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if read_u32_le(elf_bytes, off).unwrap_or(0) != PT_LOAD {
            continue;
        }

        let p_flags = read_u32_le(elf_bytes, off + 4).unwrap_or(0);
        let p_offset = read_u64_le(elf_bytes, off + 8).unwrap_or(0) as usize;
        let p_vaddr = read_u64_le(elf_bytes, off + 16).unwrap_or(0) as usize;
        let p_filesz = read_u64_le(elf_bytes, off + 32).unwrap_or(0) as usize;
        let p_memsz = read_u64_le(elf_bytes, off + 40).unwrap_or(0) as usize;
        let p_align = read_u64_le(elf_bytes, off + 48).unwrap_or(PAGE_SIZE as u64) as usize;
        let align = if p_align < PAGE_SIZE { PAGE_SIZE } else { p_align };

        if p_memsz == 0 {
            continue;
        }

        let seg_vaddr = p_vaddr.wrapping_add(bias);
        let seg_start = align_down(seg_vaddr, align);
        let seg_end = align_up(seg_vaddr + p_memsz, PAGE_SIZE);
        let map_len = seg_end - seg_start;

        let mut prot = VmProt::USER | VmProt::READ | VmProt::WRITE;
        if p_flags & 0x1 != 0 {
            prot |= VmProt::EXEC;
        }

        let req = VmMapReq {
            addr_hint: seg_start,
            len: map_len,
            prot,
            flags: VmMapFlags::FIXED | VmMapFlags::PRIVATE,
            backing: VmBacking::Anonymous { zeroed: true },
        };
        match vm_map(&req) {
            Ok(_) => {}
            Err(_) => return Err(b"vm_map failed"),
        }

        // Copy file content into mapped region.
        if p_filesz > 0 {
            let src_end = (p_offset + p_filesz).min(elf_bytes.len());
            let copy_len = src_end.saturating_sub(p_offset);
            if copy_len > 0 {
                let dst_off = seg_vaddr - seg_start;
                let dst = (seg_start + dst_off) as *mut u8;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        elf_bytes[p_offset..p_offset + copy_len].as_ptr(),
                        dst,
                        copy_len,
                    );
                }
            }
        }

        if seg_end > map_end {
            map_end = seg_end;
        }
    }

    // Parse PT_DYNAMIC to obtain symbol table info.
    let mut symtab_va = 0usize;
    let mut strtab_va = 0usize;
    let mut strsz = 0usize;
    let mut syment = 24usize;

    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if read_u32_le(elf_bytes, off).unwrap_or(0) == PT_DYNAMIC {
            let dyn_vaddr = read_u64_le(elf_bytes, off + 16).unwrap_or(0) as usize;
            let dyn_va = dyn_vaddr.wrapping_add(bias);
            let dyn_filesz = read_u64_le(elf_bytes, off + 32).unwrap_or(0) as usize;
            let entry_count = dyn_filesz / 16;
            let mut ptr = dyn_va as *const u64;
            for _ in 0..entry_count {
                let tag = unsafe { ptr.read_unaligned() } as i64;
                let val = unsafe { ptr.add(1).read_unaligned() } as usize;
                ptr = unsafe { ptr.add(2) };
                match tag {
                    t if t == DT_NULL => break,
                    t if t == DT_SYMTAB => symtab_va = val,
                    t if t == DT_STRTAB => strtab_va = val,
                    t if t == DT_STRSZ => strsz = val,
                    t if t == DT_SYMENT => syment = val,
                    _ => {}
                }
            }
            break;
        }
    }

    // Apply DT_SYMTAB/DT_STRTAB bias for ET_DYN objects (their addresses are
    // file-relative virtual addresses that need the load bias applied).
    if e_type == 3 && symtab_va != 0 {
        symtab_va = symtab_va.wrapping_add(bias);
    }
    if e_type == 3 && strtab_va != 0 {
        strtab_va = strtab_va.wrapping_add(bias);
    }

    let map_size = map_end.saturating_sub(base);

    Ok((bias, symtab_va, strtab_va, strsz, syment, base, map_size))
}

// ── Relocation processing ─────────────────────────────────────────────────────

/// Process all `Elf64_Rela` entries for a newly loaded library.
///
/// `bias` is the load bias of the object.  `global_sym_lookup` is a closure
/// that resolves a symbol name to an absolute address (used for
/// `R_X86_64_GLOB_DAT` / `R_X86_64_JUMP_SLOT`).
fn process_rela_for_handle(
    rela_va: usize,
    relasz: usize,
    relaent: usize,
    bias: usize,
    symtab_va: usize,
    strtab_va: usize,
    strsz: usize,
    syment: usize,
    lookup: &dyn Fn(&[u8]) -> Option<u64>,
) {
    if relasz == 0 || relaent == 0 {
        return;
    }
    let count = relasz / relaent;
    for i in 0..count {
        let entry = (rela_va + i * relaent) as *const u8;
        let r_offset = unsafe { read_u64_ptr(entry, 0) } as usize;
        let r_info = unsafe { read_u64_ptr(entry, 8) };
        let r_addend = unsafe { read_u64_ptr(entry, 16) } as i64;

        let r_sym = (r_info >> 32) as usize;
        let r_type = (r_info & 0xFFFF_FFFF) as u32;

        let target = r_offset.wrapping_add(bias) as *mut u64;

        let sym_addr: u64 = if r_sym != 0 && symtab_va != 0 && strtab_va != 0 {
            // Each Elf64_Sym is `syment` bytes; st_name is the first u32.
            let sym_ptr = (symtab_va + r_sym * syment) as *const u8;
            let st_name = unsafe { read_u32_ptr(sym_ptr, 0) } as usize;
            let name =
                unsafe { strtab_str(strtab_va, st_name, strsz.saturating_sub(st_name)) };
            lookup(name).unwrap_or(0)
        } else {
            0
        };

        match r_type {
            R_X86_64_NONE => {}
            R_X86_64_RELATIVE => {
                let val = (bias as i64).wrapping_add(r_addend) as u64;
                unsafe { target.write_unaligned(val) };
            }
            R_X86_64_64 => {
                let val = sym_addr.wrapping_add(r_addend as u64);
                unsafe { target.write_unaligned(val) };
            }
            R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => {
                unsafe { target.write_unaligned(sym_addr) };
            }
            R_X86_64_COPY => {
                // R_X86_64_COPY: copy `sym_addr` bytes from the shared object
                // to the target (for .bss-backed externs).  We just write the
                // address; a full implementation would copy the object data.
            }
            _ => {}
        }
    }
}

// ── Symbol lookup ─────────────────────────────────────────────────────────────

/// Look up `name` in the ELF symbol table described by `entry`.
///
/// Returns the absolute address, or `None` if not found.
pub fn lookup_in_handle(entry: &HandleEntry, name: &[u8]) -> Option<usize> {
    if entry.symtab_va == 0 || entry.strtab_va == 0 || entry.syment == 0 {
        return None;
    }
    let mut ptr = entry.symtab_va as *const u8;
    let end_guard = entry.symtab_va + 4 * 1024 * 1024;
    loop {
        let st_name = unsafe { read_u32_ptr(ptr, 0) } as usize;
        let st_info = unsafe { *ptr.add(4) };
        let st_shndx = unsafe { read_u16_ptr(ptr, 6) };
        let st_value = unsafe { read_u64_ptr(ptr, 8) };

        let bind = st_info >> 4;
        if st_shndx != 0 && (bind == STB_GLOBAL || bind == STB_WEAK) && st_value != 0 {
            let sym_name = unsafe {
                strtab_str(entry.strtab_va, st_name, entry.strsz.saturating_sub(st_name))
            };
            if sym_name == name {
                return Some(st_value.wrapping_add(entry.bias as u64) as usize);
            }
        }

        ptr = unsafe { ptr.add(entry.syment) };
        if ptr as usize > end_guard {
            break;
        }
    }
    None
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Open the shared library at `path` and return an opaque handle, or
/// `null` on failure.  Consult `dlerror()` for details.
///
/// # Safety
/// `path` must be a valid NUL-terminated C string.
pub unsafe fn dlopen(path: *const u8, _flags: i32) -> *mut core::ffi::c_void {
    if path.is_null() {
        // NULL path → return a handle representing the global namespace.
        // We use the sentinel value 1 which is never a real handle index
        // because real handles are 1-based and RTLD_NEXT is usize::MAX.
        clear_error();
        return 1usize as *mut core::ffi::c_void;
    }

    // Build a Rust str from the C string.
    let mut len = 0usize;
    while *path.add(len) != 0 {
        len += 1;
    }
    let path_bytes = core::slice::from_raw_parts(path, len);
    let path_str = match core::str::from_utf8(path_bytes) {
        Ok(s) => s,
        Err(_) => {
            set_error(b"dlopen: path is not valid UTF-8");
            return core::ptr::null_mut();
        }
    };

    dlopen_str(path_str, _flags)
}

/// Open the shared library at `path` (Rust `&str`) and return an opaque
/// handle, or `null` on failure.
pub fn dlopen_str(path: &str, _flags: i32) -> *mut core::ffi::c_void {
    let elf_bytes = match read_file(path) {
        Some(b) => b,
        None => {
            set_error(b"dlopen: cannot read library file");
            return core::ptr::null_mut();
        }
    };

    let load_size = elf_load_size(&elf_bytes).unwrap_or(4 * 1024 * 1024);
    let load_base = {
        let mut ht = HANDLES.lock();
        ht.next_lib_base(load_size)
    };

    let (bias, symtab_va, strtab_va, strsz, syment, base, map_size) =
        match map_elf(&elf_bytes, load_base) {
            Ok(v) => v,
            Err(msg) => {
                set_error(msg);
                return core::ptr::null_mut();
            }
        };

    // Process relocations.  For simplicity we resolve against already-loaded
    // handles; the caller can load dependencies first if needed.
    {
        // Find PT_DYNAMIC to get relocation info.
        if elf_bytes.len() >= 64 {
            let e_phoff = read_u64_le(&elf_bytes, 32).unwrap_or(0) as usize;
            let e_phentsize = read_u16_le(&elf_bytes, 54).unwrap_or(0) as usize;
            let e_phnum = read_u16_le(&elf_bytes, 56).unwrap_or(0) as usize;
            let e_type = read_u16_le(&elf_bytes, 16).unwrap_or(0);

            for i in 0..e_phnum {
                let off = e_phoff + i * e_phentsize;
                if read_u32_le(&elf_bytes, off).unwrap_or(0) == PT_DYNAMIC {
                    let dyn_vaddr =
                        read_u64_le(&elf_bytes, off + 16).unwrap_or(0) as usize;
                    let dyn_va = if e_type == 3 {
                        dyn_vaddr.wrapping_add(bias)
                    } else {
                        dyn_vaddr
                    };
                    let dyn_filesz =
                        read_u64_le(&elf_bytes, off + 32).unwrap_or(0) as usize;
                    let entry_count = if dyn_filesz > 0 {
                        dyn_filesz / 16
                    } else {
                        256
                    };

                    let mut rela = 0usize;
                    let mut relasz = 0usize;
                    let mut relaent = 24usize;
                    let mut jmprel = 0usize;
                    let mut pltrelsz = 0usize;

                    let mut ptr = dyn_va as *const u64;
                    for _ in 0..entry_count {
                        let tag = unsafe { ptr.read_unaligned() } as i64;
                        let val = unsafe { ptr.add(1).read_unaligned() } as usize;
                        ptr = unsafe { ptr.add(2) };
                        match tag {
                            t if t == DT_NULL => break,
                            t if t == DT_RELA => rela = val,
                            t if t == DT_RELASZ => relasz = val,
                            t if t == DT_RELAENT => relaent = val,
                            t if t == DT_JMPREL => jmprel = val,
                            t if t == DT_PLTRELSZ => pltrelsz = val,
                            _ => {}
                        }
                    }

                    // Apply bias to rela/jmprel for ET_DYN.
                    if e_type == 3 {
                        if rela != 0 { rela = rela.wrapping_add(bias); }
                        if jmprel != 0 { jmprel = jmprel.wrapping_add(bias); }
                    }

                    // Build a temporary lookup closure that searches all
                    // currently-loaded handles (snapshot taken while the
                    // lock is not held, to avoid deadlock during the reloc
                    // walk).
                    let lookup = |name: &[u8]| -> Option<u64> {
                        let ht = HANDLES.lock();
                        for slot in &ht.slots {
                            if slot.state != SLOT_USED { continue; }
                            if let Some(addr) = lookup_in_handle(slot, name) {
                                return Some(addr as u64);
                            }
                        }
                        None
                    };

                    if rela != 0 {
                        process_rela_for_handle(
                            rela, relasz, relaent, bias,
                            symtab_va, strtab_va, strsz, syment,
                            &lookup,
                        );
                    }
                    if jmprel != 0 {
                        process_rela_for_handle(
                            jmprel, pltrelsz, relaent, bias,
                            symtab_va, strtab_va, strsz, syment,
                            &lookup,
                        );
                    }
                    break;
                }
            }
        }
    }

    // Allocate a handle slot and fill it in.
    let idx = {
        let mut ht = HANDLES.lock();
        match ht.alloc() {
            Some(i) => {
                let slot = &mut ht.slots[i - 1];
                slot.bias = bias;
                slot.symtab_va = symtab_va;
                slot.strtab_va = strtab_va;
                slot.strsz = strsz;
                slot.syment = syment;
                slot.base = base;
                slot.map_size = map_size;
                i
            }
            None => {
                set_error(b"dlopen: handle table full");
                return core::ptr::null_mut();
            }
        }
    };

    clear_error();
    idx as *mut core::ffi::c_void
}

/// Resolve symbol `name` in the library identified by `handle`.
///
/// If `handle` is `RTLD_DEFAULT` (null), all currently-open handles are
/// searched in load order.
///
/// Returns the symbol's absolute address, or `null` on failure.
///
/// # Safety
/// `name` must be a valid NUL-terminated C string.
pub unsafe fn dlsym(
    handle: *mut core::ffi::c_void,
    name: *const u8,
) -> *mut core::ffi::c_void {
    if name.is_null() {
        set_error(b"dlsym: null symbol name");
        return core::ptr::null_mut();
    }

    let mut len = 0usize;
    while *name.add(len) != 0 {
        len += 1;
    }
    let sym_name = core::slice::from_raw_parts(name, len);

    dlsym_bytes(handle, sym_name)
}

/// Resolve symbol `name` (as a byte slice) in the library identified by
/// `handle`.
pub fn dlsym_bytes(
    handle: *mut core::ffi::c_void,
    name: &[u8],
) -> *mut core::ffi::c_void {
    let ht = HANDLES.lock();

    if handle == RTLD_DEFAULT || handle == RTLD_NEXT {
        // Search all loaded handles in slot order.
        for slot in &ht.slots {
            if slot.state != SLOT_USED {
                continue;
            }
            if let Some(addr) = lookup_in_handle(slot, name) {
                clear_error();
                return addr as *mut core::ffi::c_void;
            }
        }
        set_error(b"dlsym: symbol not found");
        return core::ptr::null_mut();
    }

    // Handle-specific lookup.
    let idx = handle as usize;
    if idx == 0 || idx > MAX_HANDLES {
        set_error(b"dlsym: invalid handle");
        return core::ptr::null_mut();
    }
    let slot = &ht.slots[idx - 1];
    if slot.state != SLOT_USED {
        set_error(b"dlsym: handle not open");
        return core::ptr::null_mut();
    }
    match lookup_in_handle(slot, name) {
        Some(addr) => {
            clear_error();
            addr as *mut core::ffi::c_void
        }
        None => {
            set_error(b"dlsym: symbol not found");
            core::ptr::null_mut()
        }
    }
}

/// Release a handle returned by `dlopen`.
///
/// Returns `0` on success, `-1` on error.
pub fn dlclose(handle: *mut core::ffi::c_void) -> i32 {
    if handle == RTLD_DEFAULT {
        set_error(b"dlclose: cannot close RTLD_DEFAULT");
        return -1;
    }
    let idx = handle as usize;
    if idx == 0 || idx > MAX_HANDLES {
        set_error(b"dlclose: invalid handle");
        return -1;
    }
    let mut ht = HANDLES.lock();
    if ht.slots[idx - 1].state != SLOT_USED {
        set_error(b"dlclose: handle not open");
        return -1;
    }
    ht.free(idx);
    clear_error();
    0
}

/// Return a pointer to the most recent error string, then clear the pending
/// error.  Returns `null` if no error has occurred since the last call.
///
/// The returned pointer is valid until the next call to any `dl*` function.
pub fn dlerror() -> *const u8 {
    let mut state = ERR_STATE.lock();
    if !state.pending {
        return core::ptr::null();
    }
    // Copy into the stable output buffer via a raw pointer to avoid the
    // `static_mut_refs` lint.
    let buf_ptr = core::ptr::addr_of_mut!(DLERROR_BUF);
    unsafe { (*buf_ptr).copy_from_slice(&state.buf) };
    state.clear();
    buf_ptr as *const u8
}
