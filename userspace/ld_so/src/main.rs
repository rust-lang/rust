//! `ld_so` — Dynamic ELF loader for Janix.
//!
//! This binary serves as the runtime dynamic linker / loader (`ld.so` /
//! `ld-linux.so` equivalent) for the Janix userspace.
//!
//! ## When does this run?
//!
//! When the kernel `exec`s an ELF binary that contains a `PT_INTERP` segment
//! pointing to this binary (`/lib/ld.so`), the kernel:
//!
//! 1. Maps the main executable's `PT_LOAD` segments into the new address
//!    space.
//! 2. Maps this interpreter into the address space at `AT_BASE`
//!    (currently `0x7F00_0000`).
//! 3. Sets the initial instruction pointer to this interpreter's entry point.
//! 4. Places the following `AT_*` entries in the auxiliary vector:
//!    - `AT_PHDR`  — virtual address of the main program's program-header table.
//!    - `AT_PHENT` — size of one program-header entry (56 for ELF64).
//!    - `AT_PHNUM` — number of program-header entries.
//!    - `AT_ENTRY` — real entry point of the main executable.
//!    - `AT_BASE`  — load base of this interpreter.
//!    - `AT_PAGESZ` — system page size.
//!
//! ## What this loader does
//!
//! 1. Read the auxv to obtain `AT_PHDR`, `AT_PHNUM`, `AT_ENTRY`, `AT_BASE`.
//! 2. Walk the main executable's program headers to find `PT_DYNAMIC`.
//! 3. Parse `PT_DYNAMIC` to find `DT_NEEDED` entries.
//! 4. For each needed library open it from the VFS, map its `PT_LOAD`
//!    segments into the current address space using `SYS_VM_MAP`, and
//!    process its relocations.
//! 5. Process the main executable's relocations.
//! 6. Call `DT_INIT` / `DT_INIT_ARRAY` functions.
//! 7. Jump to `AT_ENTRY`.
//!
//! ## Design notes
//!
//! - No heap is used for the global symbol table; a fixed-capacity open-addressing
//!   hash table backed by a static array provides O(1) lookups without
//!   dynamic allocation.
//! - Library search path is `/lib` followed by `/usr/lib`.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use alloc::vec::Vec;
use abi::auxv;
use abi::vm::{VmBacking, VmMapFlags, VmProt};
use stem::println;
use stem::syscall::{auxv_get, exit, vfs_close, vfs_open, vfs_read, vfs_seek, vfs_stat};
use stem::vm::{VmMapReq, vm_map};

// ── Auxiliary vector helpers ─────────────────────────────────────────────────

/// Parse the serialised auxv blob returned by `SYS_AUXV_GET`.
///
/// Format: `count: u32 LE` followed by `count` × `(type: u64 LE, value: u64 LE)`.
fn parse_auxv(buf: &[u8]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    if buf.len() < 4 {
        return out;
    }
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut off = 4usize;
    for _ in 0..count {
        if off + 16 > buf.len() {
            break;
        }
        let typ = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        let val = u64::from_le_bytes(buf[off + 8..off + 16].try_into().unwrap());
        out.push((typ, val));
        off += 16;
    }
    out
}

fn auxv_find(entries: &[(u64, u64)], typ: u64) -> u64 {
    entries.iter().find(|&&(k, _)| k == typ).map(|&(_, v)| v).unwrap_or(0)
}

// ── ELF parsing helpers (no_std, bare-metal) ─────────────────────────────────

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

fn write_u64_le(buf: &mut [u8], off: usize, val: u64) {
    if off + 8 <= buf.len() {
        buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
    }
}

// ── ELF constants ────────────────────────────────────────────────────────────

// Program-header types
const PT_LOAD:    u32 = 1;
const PT_DYNAMIC: u32 = 2;

// Dynamic-section tags
const DT_NULL:       i64 = 0;
const DT_NEEDED:     i64 = 1;
const DT_PLTRELSZ:   i64 = 2;
const DT_PLTGOT:     i64 = 3;
const DT_HASH:       i64 = 4;
const DT_STRTAB:     i64 = 5;
const DT_SYMTAB:     i64 = 6;
const DT_RELA:       i64 = 7;
const DT_RELASZ:     i64 = 8;
const DT_RELAENT:    i64 = 9;
const DT_STRSZ:      i64 = 10;
const DT_SYMENT:     i64 = 11;
const DT_INIT:       i64 = 12;
const DT_FINI:       i64 = 13;
const DT_JMPREL:     i64 = 23;
const DT_BIND_NOW:   i64 = 24;
const DT_INIT_ARRAY: i64 = 25;
const DT_FINI_ARRAY: i64 = 26;
const DT_INIT_ARRAYSZ: i64 = 27;
const DT_FINI_ARRAYSZ: i64 = 28;
const DT_FLAGS_1:    i64 = 0x6ffffffb_u32 as i64;

// Relocation types (x86_64 / R_X86_64_*)
const R_X86_64_NONE:      u32 = 0;
const R_X86_64_64:        u32 = 1;
const R_X86_64_RELATIVE:  u32 = 8;
const R_X86_64_GLOB_DAT:  u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_COPY:      u32 = 5;

// ELF symbol bind
const STB_LOCAL:  u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK:   u8 = 2;

// ── Loaded-object representation ─────────────────────────────────────────────

/// A single ELF shared object or executable that has been mapped into the
/// current address space.
struct LoadedObject {
    /// Load bias: the difference between the virtual address in the ELF file
    /// and the actual virtual address in the running process.
    ///
    /// For PIE / ET_DYN objects this equals the base address they were mapped
    /// at.  For ET_EXEC objects it is normally zero.
    bias: u64,

    /// Virtual address of the `.dynamic` section (after biasing), or 0.
    dynamic_va: u64,

    /// Base virtual address where this object was mapped.
    base: u64,
}

// ── Global symbol table ───────────────────────────────────────────────────────

/// Maximum number of symbols we can track.
const MAX_SYMBOLS: usize = 4096;

struct SymEntry {
    /// 64-bit FNV-1a hash of the symbol name.  Zero means empty slot.
    hash: u64,
    /// Absolute virtual address of the symbol.
    addr: u64,
    /// Symbol name, NUL-padded.
    name: [u8; 64],
}

impl SymEntry {
    const fn empty() -> Self {
        SymEntry { hash: 0, addr: 0, name: [0u8; 64] }
    }
}

struct SymbolTable {
    entries: [SymEntry; MAX_SYMBOLS],
    count: usize,
}

impl SymbolTable {
    const fn new() -> Self {
        // Can't use array-repeat expressions for non-Copy types in const
        // contexts on stable; hand-initialise.
        const E: SymEntry = SymEntry::empty();
        SymbolTable {
            entries: [E; MAX_SYMBOLS],
            count: 0,
        }
    }

    fn fnv1a(name: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce4_84222325;
        for &b in name {
            if b == 0 { break; }
            h ^= b as u64;
            h = h.wrapping_mul(0x00000100_000001b3);
        }
        h | 1 // ensure non-zero so we can use 0 as sentinel
    }

    /// Insert `(name, addr)` into the table.  Silently drops on overflow.
    fn insert(&mut self, name: &[u8], addr: u64) {
        let h = Self::fnv1a(name);
        let mut idx = (h as usize) % MAX_SYMBOLS;
        for _ in 0..MAX_SYMBOLS {
            if self.entries[idx].hash == 0 {
                self.entries[idx].hash = h;
                self.entries[idx].addr = addr;
                let copy_len = name.len().min(63);
                self.entries[idx].name[..copy_len].copy_from_slice(&name[..copy_len]);
                self.entries[idx].name[copy_len] = 0;
                self.count += 1;
                return;
            }
            // Already have this symbol (prefer first definition).
            if self.entries[idx].hash == h && name_eq(&self.entries[idx].name, name) {
                return;
            }
            idx = (idx + 1) % MAX_SYMBOLS;
        }
        // Table full — silently drop.
    }

    fn lookup(&self, name: &[u8]) -> Option<u64> {
        let h = Self::fnv1a(name);
        let mut idx = (h as usize) % MAX_SYMBOLS;
        for _ in 0..MAX_SYMBOLS {
            if self.entries[idx].hash == 0 {
                return None;
            }
            if self.entries[idx].hash == h && name_eq(&self.entries[idx].name, name) {
                return Some(self.entries[idx].addr);
            }
            idx = (idx + 1) % MAX_SYMBOLS;
        }
        None
    }
}

fn name_eq(stored: &[u8; 64], name: &[u8]) -> bool {
    for (i, &b) in name.iter().enumerate() {
        if b == 0 { break; }
        if i >= 64 || stored[i] != b { return false; }
    }
    // ensure the stored name ends at the same position
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    end >= 64 || stored[end] == 0
}

// Global symbol table — initialised once at loader entry.
static mut SYMTAB: SymbolTable = SymbolTable::new();

fn sym_insert(name: &[u8], addr: u64) {
    unsafe { SYMTAB.insert(name, addr) }
}

fn sym_lookup(name: &[u8]) -> Option<u64> {
    unsafe { SYMTAB.lookup(name) }
}

// ── File I/O helpers ──────────────────────────────────────────────────────────

/// Read up to `buf.len()` bytes from an open fd at file offset `offset`.
fn pread(fd: u32, buf: &mut [u8], offset: u64) -> usize {
    use stem::syscall::vfs_seek;
    let _ = vfs_seek(fd, offset as i64, 0 /* SEEK_SET */);
    match vfs_read(fd, buf) {
        Ok(n) => n,
        Err(_) => 0,
    }
}

/// Read the entire file identified by `path` into a newly-allocated `Vec`.
fn read_file(path: &str) -> Option<Vec<u8>> {
    let fd = vfs_open(path, 0 /* O_RDONLY */).ok()?;
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
        if n == 0 { break; }
        pos += n;
    }
    let _ = vfs_close(fd);
    if pos < size { None } else { Some(buf) }
}

// ── ELF loading ───────────────────────────────────────────────────────────────

const PAGE_SIZE: usize = 4096;

fn align_down(v: usize, a: usize) -> usize { v & !(a - 1) }
fn align_up(v: usize, a: usize) -> usize { (v + a - 1) & !(a - 1) }

/// Map all PT_LOAD segments of `elf_bytes` into the current address space
/// starting at `load_base`.  Returns the effective load bias (for PIE) or
/// zero for ET_EXEC.
///
/// # Safety
/// Writes to the current process address space via `vm_map`.
fn map_elf_segments(elf_bytes: &[u8], load_base: usize) -> Option<usize> {
    if elf_bytes.len() < 64 { return None; }
    if &elf_bytes[0..4] != b"\x7fELF" { return None; }
    if elf_bytes[4] != 2 || elf_bytes[5] != 1 { return None; } // ELF64 LE only

    let e_type    = read_u16_le(elf_bytes, 16)?; // ET_DYN == 3, ET_EXEC == 2
    let e_phoff   = read_u64_le(elf_bytes, 32)? as usize;
    let e_phentsize = read_u16_le(elf_bytes, 54)? as usize;
    let e_phnum   = read_u16_le(elf_bytes, 56)? as usize;
    if e_phentsize == 0 || e_phnum == 0 { return None; }

    // Determine the minimum virtual address across all PT_LOAD segments
    // so we can compute the bias for PIE objects.
    let mut min_vaddr = usize::MAX;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        let p_type = read_u32_le(elf_bytes, off)?;
        if p_type == PT_LOAD {
            let p_vaddr = read_u64_le(elf_bytes, off + 16)? as usize;
            if p_vaddr < min_vaddr { min_vaddr = p_vaddr; }
        }
    }
    if min_vaddr == usize::MAX { return None; }

    // For ET_DYN, bias = load_base - min_vaddr.
    // For ET_EXEC, we ignore load_base and load at the fixed vaddrs.
    let bias: usize = if e_type == 3 /* ET_DYN */ {
        load_base.wrapping_sub(min_vaddr)
    } else {
        0
    };

    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        let p_type   = read_u32_le(elf_bytes, off)?;
        if p_type != PT_LOAD { continue; }

        let p_flags  = read_u32_le(elf_bytes, off + 4)?;
        let p_offset = read_u64_le(elf_bytes, off + 8)? as usize;
        let p_vaddr  = read_u64_le(elf_bytes, off + 16)? as usize;
        let p_filesz = read_u64_le(elf_bytes, off + 32)? as usize;
        let p_memsz  = read_u64_le(elf_bytes, off + 40)? as usize;
        let p_align  = read_u64_le(elf_bytes, off + 48)? as usize;
        let align    = if p_align < PAGE_SIZE { PAGE_SIZE } else { p_align };

        if p_memsz == 0 { continue; }

        let seg_vaddr = p_vaddr.wrapping_add(bias);
        let seg_start = align_down(seg_vaddr, align);
        let seg_end   = align_up(seg_vaddr + p_memsz, PAGE_SIZE);
        let map_len   = seg_end - seg_start;

        // Build protection flags
        let mut prot = VmProt::USER;
        if p_flags & 0x4 != 0 { prot |= VmProt::READ; }
        if p_flags & 0x2 != 0 { prot |= VmProt::WRITE; }
        if p_flags & 0x1 != 0 { prot |= VmProt::EXEC; }
        // Need write to copy data in; we'll re-protect after if needed.
        let map_prot = prot | VmProt::READ | VmProt::WRITE;

        let req = VmMapReq {
            addr_hint: seg_start,
            len: map_len,
            prot: map_prot,
            flags: VmMapFlags::FIXED | VmMapFlags::PRIVATE,
            backing: VmBacking::Anonymous { zeroed: true },
        };
        let resp = match vm_map(&req) {
            Ok(r) => r,
            Err(e) => {
                println!("[ld.so] vm_map failed for segment at 0x{:x}: {:?}", seg_start, e);
                return None;
            }
        };

        // Copy file data into the mapped region.
        if p_filesz > 0 {
            let src_end = (p_offset + p_filesz).min(elf_bytes.len());
            let copy_len = src_end.saturating_sub(p_offset);
            if copy_len > 0 {
                let dst_off = seg_vaddr - seg_start;
                let dst = resp.addr + dst_off;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        elf_bytes[p_offset..p_offset + copy_len].as_ptr(),
                        dst as *mut u8,
                        copy_len,
                    );
                }
            }
        }
        // BSS is already zeroed from the anonymous mapping.
    }

    Some(bias)
}

/// Find the virtual address (already biased) of PT_DYNAMIC in a loaded ELF.
fn find_dynamic_va(elf_bytes: &[u8], bias: usize) -> Option<usize> {
    let e_phoff     = read_u64_le(elf_bytes, 32)? as usize;
    let e_phentsize = read_u16_le(elf_bytes, 54)? as usize;
    let e_phnum     = read_u16_le(elf_bytes, 56)? as usize;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if read_u32_le(elf_bytes, off)? == PT_DYNAMIC {
            let vaddr = read_u64_le(elf_bytes, off + 16)? as usize;
            return Some(vaddr.wrapping_add(bias));
        }
    }
    None
}

/// Read the `PT_DYNAMIC` section (at virtual address `dynamic_va`) and return
/// the relevant `DT_*` values.
struct DynInfo {
    strtab:      usize, // virtual address of DT_STRTAB
    strsz:       usize,
    symtab:      usize, // virtual address of DT_SYMTAB
    syment:      usize, // size of one Elf64_Sym entry (usually 24)
    rela:        usize, // DT_RELA
    relasz:      usize, // DT_RELASZ
    relaent:     usize, // DT_RELAENT (usually 24)
    jmprel:      usize, // DT_JMPREL
    pltrelsz:    usize, // DT_PLTRELSZ
    init:        usize,
    fini:        usize,
    init_array:  usize,
    init_arraysz:usize,
    needed:      Vec<usize>, // offsets into strtab for each DT_NEEDED
}

fn read_dyn_info(dynamic_va: usize) -> DynInfo {
    let mut di = DynInfo {
        strtab: 0, strsz: 0, symtab: 0, syment: 24,
        rela: 0, relasz: 0, relaent: 24,
        jmprel: 0, pltrelsz: 0,
        init: 0, fini: 0, init_array: 0, init_arraysz: 0,
        needed: Vec::new(),
    };

    let mut ptr = dynamic_va as *const u64;
    loop {
        let tag  = unsafe { ptr.read_unaligned() } as i64;
        let val  = unsafe { ptr.add(1).read_unaligned() } as usize;
        ptr = unsafe { ptr.add(2) };

        match tag {
            t if t == DT_NULL       => break,
            t if t == DT_STRTAB     => di.strtab = val,
            t if t == DT_STRSZ      => di.strsz  = val,
            t if t == DT_SYMTAB     => di.symtab = val,
            t if t == DT_SYMENT     => di.syment = val,
            t if t == DT_RELA       => di.rela    = val,
            t if t == DT_RELASZ     => di.relasz  = val,
            t if t == DT_RELAENT    => di.relaent = val,
            t if t == DT_JMPREL     => di.jmprel  = val,
            t if t == DT_PLTRELSZ   => di.pltrelsz = val,
            t if t == DT_INIT       => di.init      = val,
            t if t == DT_FINI       => di.fini      = val,
            t if t == DT_INIT_ARRAY => di.init_array = val,
            t if t == DT_INIT_ARRAYSZ => di.init_arraysz = val,
            t if t == DT_NEEDED     => di.needed.push(val),
            _ => {}
        }
    }
    di
}

/// Read a NUL-terminated string from `strtab_va + offset`.
///
/// Returns a slice into the string table memory.  Guaranteed not to exceed
/// `max_len` bytes.
fn strtab_str(strtab_va: usize, offset: usize, max_len: usize) -> &'static [u8] {
    let ptr = (strtab_va + offset) as *const u8;
    let mut len = 0;
    while len < max_len {
        if unsafe { ptr.add(len).read() } == 0 { break; }
        len += 1;
    }
    unsafe { core::slice::from_raw_parts(ptr, len) }
}

// ── Symbol lookup in a single shared object ───────────────────────────────────

/// Look up `name` in the dynamic symbol table of a single loaded ELF.
///
/// `symtab_va` and `strtab_va` are absolute virtual addresses (already biased).
fn lookup_in_symtab(
    name: &[u8],
    symtab_va: usize,
    strtab_va: usize,
    strsz: usize,
    syment: usize,
    bias: usize,
) -> Option<u64> {
    // Walk the symbol table linearly.  A proper implementation would use
    // the ELF hash table (DT_HASH / DT_GNU_HASH) but the linear scan is
    // correct and simple enough for the initial implementation.
    let mut ptr = symtab_va as *const u8;
    loop {
        // Elf64_Sym: st_name(4) st_info(1) st_other(1) st_shndx(2) st_value(8) st_size(8)
        let st_name  = unsafe { read_u32_from_ptr(ptr, 0) } as usize;
        let st_info  = unsafe { ptr.add(4).read() };
        let st_shndx = unsafe { read_u16_from_ptr(ptr, 6) };
        let st_value = unsafe { read_u64_from_ptr(ptr, 8) };

        let bind = st_info >> 4;
        // SHN_UNDEF == 0; skip undefined symbols.
        if st_shndx != 0 && (bind == STB_GLOBAL || bind == STB_WEAK) && st_value != 0 {
            let sym_name = strtab_str(strtab_va, st_name, strsz.saturating_sub(st_name));
            if sym_name == name {
                return Some(st_value.wrapping_add(bias as u64));
            }
        }

        // Stop at the first all-zero entry (heuristic: no name, no value).
        // A proper termination would use the hash table's nchain count.
        if st_name == 0 && st_value == 0 && bind == STB_LOCAL {
            // Could be a valid local sym or the end sentinel — continue.
        }
        // Advance to next entry.
        ptr = unsafe { ptr.add(syment) };

        // Safeguard: if we've wandered past a reasonable range, stop.
        if ptr as usize > symtab_va + 4 * 1024 * 1024 {
            break;
        }
    }
    None
}

unsafe fn read_u16_from_ptr(p: *const u8, off: usize) -> u16 {
    u16::from_le_bytes([*p.add(off), *p.add(off + 1)])
}

unsafe fn read_u32_from_ptr(p: *const u8, off: usize) -> u32 {
    u32::from_le_bytes([*p.add(off), *p.add(off + 1), *p.add(off + 2), *p.add(off + 3)])
}

unsafe fn read_u64_from_ptr(p: *const u8, off: usize) -> u64 {
    u64::from_le_bytes([
        *p.add(off), *p.add(off+1), *p.add(off+2), *p.add(off+3),
        *p.add(off+4), *p.add(off+5), *p.add(off+6), *p.add(off+7),
    ])
}

// ── Relocation processing ─────────────────────────────────────────────────────

/// Process all Elf64_Rela entries in `[rela_va, rela_va+relasz)`.
///
/// `bias` is the load bias of the object being relocated.
/// `symtab_va`, `strtab_va`, `strsz`, `syment` describe the dynamic symbol
/// table so we can resolve symbolic relocations.
fn process_rela(
    rela_va: usize,
    relasz: usize,
    relaent: usize,
    bias: usize,
    symtab_va: usize,
    strtab_va: usize,
    strsz: usize,
    syment: usize,
) {
    if relasz == 0 || relaent == 0 { return; }
    let count = relasz / relaent;
    for i in 0..count {
        let entry = (rela_va + i * relaent) as *const u8;
        // Elf64_Rela: r_offset(8) r_info(8) r_addend(8)
        let r_offset = unsafe { read_u64_from_ptr(entry, 0) } as usize;
        let r_info   = unsafe { read_u64_from_ptr(entry, 8) };
        let r_addend = unsafe { read_u64_from_ptr(entry, 16) } as i64;

        let r_sym  = (r_info >> 32) as usize;
        let r_type = (r_info & 0xFFFF_FFFF) as u32;

        let target = (r_offset.wrapping_add(bias)) as *mut u64;

        match r_type {
            R_X86_64_NONE => {}

            R_X86_64_RELATIVE => {
                // *target = bias + addend
                let val = (bias as i64).wrapping_add(r_addend) as u64;
                unsafe { target.write_unaligned(val); }
            }

            R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT | R_X86_64_64 => {
                if r_sym == 0 {
                    // No symbol — treat as relative.
                    let val = (bias as i64).wrapping_add(r_addend) as u64;
                    unsafe { target.write_unaligned(val); }
                    continue;
                }
                // Resolve symbol.
                let sym_ptr = (symtab_va + r_sym * syment) as *const u8;
                let st_name = unsafe { read_u32_from_ptr(sym_ptr, 0) } as usize;
                let name = strtab_str(strtab_va, st_name, strsz.saturating_sub(st_name));

                let sym_val = sym_lookup(name).or_else(|| {
                    // Try the local symbol table as a fallback.
                    lookup_in_symtab(name, symtab_va, strtab_va, strsz, syment, bias)
                });

                if let Some(addr) = sym_val {
                    let val = (addr as i64).wrapping_add(r_addend) as u64;
                    unsafe { target.write_unaligned(val); }
                } else {
                    println!(
                        "[ld.so] warning: unresolved symbol '{}'",
                        core::str::from_utf8(name).unwrap_or("?")
                    );
                }
            }

            R_X86_64_COPY => {
                // Copy data from the shared-library's symbol into the BSS
                // placeholder in the executable.  Rarely used in modern code.
                // We simply zero the target for now.
                unsafe { target.write_unaligned(0); }
            }

            _ => {
                // Unknown relocation type — skip silently.
            }
        }
    }
}

// ── Library loading ───────────────────────────────────────────────────────────

/// Library search paths (tried in order).
const SEARCH_PATHS: &[&str] = &["/lib", "/usr/lib"];

/// Mutable base address counter for dynamically-loaded libraries.
/// Each library is placed at a new non-overlapping base.
static mut NEXT_LIB_BASE: usize = 0x6000_0000;

fn next_lib_base(map_size: usize) -> usize {
    let base = unsafe { NEXT_LIB_BASE };
    unsafe { NEXT_LIB_BASE = NEXT_LIB_BASE + align_up(map_size, PAGE_SIZE) + PAGE_SIZE; }
    base
}

/// Load a single shared library by `soname`, map it, populate the global symbol
/// table, and return its load bias.
///
/// Returns `None` if the library cannot be found or loaded.
fn load_library(soname: &[u8]) -> Option<usize> {
    // Build path strings and try each search directory.
    let name_str = core::str::from_utf8(soname).ok()?;

    for &dir in SEARCH_PATHS {
        // We need a small stack-allocated path buffer.
        let mut path = [0u8; 256];
        let dlen = dir.len();
        let nlen = name_str.len();
        if dlen + 1 + nlen + 1 > 255 { continue; }
        path[..dlen].copy_from_slice(dir.as_bytes());
        path[dlen] = b'/';
        path[dlen + 1..dlen + 1 + nlen].copy_from_slice(soname);
        path[dlen + 1 + nlen] = 0;

        let path_str = core::str::from_utf8(&path[..dlen + 1 + nlen]).ok()?;

        if let Some(bytes) = read_file(path_str) {
            let load_size = elf_load_size(&bytes).unwrap_or(4 * 1024 * 1024);
            let base = next_lib_base(load_size);

            if let Some(bias) = map_elf_segments(&bytes, base) {
                // Populate the global symbol table from this library's exports.
                if let Some(dyn_va) = find_dynamic_va(&bytes, bias) {
                    let di = read_dyn_info(dyn_va);
                    if di.symtab != 0 && di.strtab != 0 {
                        export_symbols(di.symtab, di.strtab, di.strsz, di.syment, bias);
                    }

                    // Process the library's own relocations.
                    if di.rela != 0 {
                        process_rela(
                            di.rela, di.relasz, di.relaent,
                            bias, di.symtab, di.strtab, di.strsz, di.syment,
                        );
                    }
                    if di.jmprel != 0 {
                        process_rela(
                            di.jmprel, di.pltrelsz, di.relaent,
                            bias, di.symtab, di.strtab, di.strsz, di.syment,
                        );
                    }

                    // Recursively load the library's own DT_NEEDED entries.
                    for &off in &di.needed {
                        if di.strtab != 0 {
                            let dep = strtab_str(di.strtab, off, di.strsz.saturating_sub(off));
                            if !dep.is_empty() {
                                load_library(dep);
                            }
                        }
                    }
                }
                println!("[ld.so] loaded '{}' at base 0x{:x} (bias 0x{:x})", name_str, base, bias);
                return Some(bias);
            }
        }
    }
    println!("[ld.so] warning: library '{}' not found", name_str);
    None
}

/// Export all global symbols from a loaded DSO into the global symbol table.
fn export_symbols(symtab_va: usize, strtab_va: usize, strsz: usize, syment: usize, bias: usize) {
    let mut ptr = symtab_va as *const u8;
    let end_heuristic = symtab_va + 1 * 1024 * 1024; // 1 MiB safeguard

    loop {
        let st_name  = unsafe { read_u32_from_ptr(ptr, 0) } as usize;
        let st_info  = unsafe { ptr.add(4).read() };
        let st_shndx = unsafe { read_u16_from_ptr(ptr, 6) };
        let st_value = unsafe { read_u64_from_ptr(ptr, 8) };

        let bind = st_info >> 4;
        if st_shndx != 0 && (bind == STB_GLOBAL || bind == STB_WEAK) && st_value != 0 {
            let name = strtab_str(strtab_va, st_name, strsz.saturating_sub(st_name));
            if !name.is_empty() {
                let addr = st_value.wrapping_add(bias as u64);
                sym_insert(name, addr);
            }
        }

        ptr = unsafe { ptr.add(syment) };
        if ptr as usize > end_heuristic { break; }
    }
}

/// Compute the total virtual address span of all PT_LOAD segments, which gives
/// an upper bound on how much memory the ELF will use.
fn elf_load_size(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 64 { return None; }
    let e_phoff     = read_u64_le(bytes, 32)? as usize;
    let e_phentsize = read_u16_le(bytes, 54)? as usize;
    let e_phnum     = read_u16_le(bytes, 56)? as usize;
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if read_u32_le(bytes, off)? == PT_LOAD {
            let vaddr = read_u64_le(bytes, off + 16)? as usize;
            let memsz = read_u64_le(bytes, off + 40)? as usize;
            if vaddr < lo { lo = vaddr; }
            if vaddr + memsz > hi { hi = vaddr + memsz; }
        }
    }
    if hi > lo { Some(hi - lo) } else { None }
}

// ── Init / fini arrays ────────────────────────────────────────────────────────

/// Call all functions in the `DT_INIT_ARRAY`.
fn run_init_array(init_array_va: usize, arraysz: usize) {
    if init_array_va == 0 || arraysz == 0 { return; }
    let count = arraysz / 8;
    for i in 0..count {
        let fn_ptr_va = (init_array_va + i * 8) as *const usize;
        let fn_addr = unsafe { fn_ptr_va.read_unaligned() };
        if fn_addr != 0 && fn_addr != usize::MAX {
            let f: extern "C" fn() = unsafe { core::mem::transmute(fn_addr) };
            f();
        }
    }
}

/// Call a single DT_INIT function if its address is non-zero.
fn run_init(init_va: usize) {
    if init_va == 0 { return; }
    let f: extern "C" fn() = unsafe { core::mem::transmute(init_va) };
    f();
}

// ── Loader entry point ────────────────────────────────────────────────────────

#[stem::main]
fn main(_arg: usize) -> ! {
    println!("[ld.so] dynamic loader starting");

    // ── Step 1: Read the auxiliary vector ────────────────────────────────────
    let needed_size = auxv_get(&mut []).unwrap_or(4);
    let mut auxv_buf = alloc::vec![0u8; needed_size];
    let _ = auxv_get(&mut auxv_buf);
    let auxv_entries = parse_auxv(&auxv_buf);

    let at_phdr  = auxv_find(&auxv_entries, auxv::AT_PHDR)  as usize;
    let at_phent = auxv_find(&auxv_entries, auxv::AT_PHENT) as usize;
    let at_phnum = auxv_find(&auxv_entries, auxv::AT_PHNUM) as usize;
    let at_entry = auxv_find(&auxv_entries, auxv::AT_ENTRY) as usize;
    let at_base  = auxv_find(&auxv_entries, auxv::AT_BASE)  as usize;

    println!(
        "[ld.so] AT_PHDR=0x{:x}  AT_PHNUM={}  AT_ENTRY=0x{:x}  AT_BASE=0x{:x}",
        at_phdr, at_phnum, at_entry, at_base
    );

    if at_phdr == 0 || at_phnum == 0 || at_entry == 0 {
        println!("[ld.so] error: missing required AT_* entries — aborting");
        exit(1);
    }

    // ── Step 2: Walk the main executable's program headers ───────────────────
    // Find PT_DYNAMIC and infer the main executable's load bias from AT_PHDR.
    //
    // For a PIE executable, AT_PHDR = actual_phdr_va, so the bias is:
    //   bias = AT_PHDR - phoff_in_file
    // But we don't have the file offset here.  Instead we'll find the PT_DYNAMIC
    // virtual address from the phdrs (which are already biased/resolved).
    let mut dynamic_va: usize = 0;
    let mut main_exec_bias: usize = 0; // best-effort; 0 for ET_EXEC
    for i in 0..at_phnum {
        let off = at_phdr + i * at_phent;
        let p_type  = unsafe { read_u32_le_from_va(off, 0) };
        let p_vaddr = unsafe { read_u64_le_from_va(off, 16) } as usize;
        if p_type == PT_DYNAMIC {
            dynamic_va = p_vaddr;
        }
    }

    if dynamic_va == 0 {
        // No PT_DYNAMIC → statically linked executable; just jump to entry.
        println!("[ld.so] no PT_DYNAMIC found; treating as static — jumping to entry");
        jump_to_entry(at_entry);
    }

    println!("[ld.so] PT_DYNAMIC at 0x{:x}", dynamic_va);

    // ── Step 3: Parse PT_DYNAMIC ──────────────────────────────────────────────
    let di = read_dyn_info(dynamic_va);

    // ── Step 4: Export main executable's symbols first ───────────────────────
    if di.symtab != 0 && di.strtab != 0 {
        export_symbols(di.symtab, di.strtab, di.strsz, di.syment, main_exec_bias);
    }

    // ── Step 5: Load DT_NEEDED libraries ──────────────────────────────────────
    for &off in &di.needed {
        if di.strtab != 0 {
            let soname = strtab_str(di.strtab, off, di.strsz.saturating_sub(off));
            if !soname.is_empty() {
                println!(
                    "[ld.so] loading DT_NEEDED: '{}'",
                    core::str::from_utf8(soname).unwrap_or("?")
                );
                load_library(soname);
            }
        }
    }

    // ── Step 6: Relocate the main executable ──────────────────────────────────
    if di.rela != 0 {
        process_rela(
            di.rela, di.relasz, di.relaent,
            main_exec_bias, di.symtab, di.strtab, di.strsz, di.syment,
        );
    }
    if di.jmprel != 0 {
        process_rela(
            di.jmprel, di.pltrelsz, di.relaent,
            main_exec_bias, di.symtab, di.strtab, di.strsz, di.syment,
        );
    }

    // ── Step 7: Run init functions ────────────────────────────────────────────
    run_init(di.init);
    run_init_array(di.init_array, di.init_arraysz);

    // ── Step 8: Transfer control to the main entry point ─────────────────────
    println!("[ld.so] jumping to entry point 0x{:x}", at_entry);
    jump_to_entry(at_entry);
}

/// Jump to the real executable entry point, never returning.
fn jump_to_entry(entry: usize) -> ! {
    let f: extern "C" fn() -> ! = unsafe { core::mem::transmute(entry) };
    f()
}

/// Read a u32 from a virtual address (already mapped into our address space).
unsafe fn read_u32_le_from_va(base: usize, off: usize) -> u32 {
    let p = (base + off) as *const u8;
    u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)])
}

/// Read a u64 from a virtual address.
unsafe fn read_u64_le_from_va(base: usize, off: usize) -> u64 {
    let p = (base + off) as *const u8;
    u64::from_le_bytes([
        *p, *p.add(1), *p.add(2), *p.add(3),
        *p.add(4), *p.add(5), *p.add(6), *p.add(7),
    ])
}
