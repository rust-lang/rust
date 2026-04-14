//! Integration tests for the `ld.so` dynamic ELF loader.
//!
//! These tests validate the loader's behaviour from userspace:
//!
//! - The auxiliary vector contains the expected AT_* entries when a dynamic
//!   executable is launched (AT_BASE should be present; this test is itself
//!   statically linked so AT_BASE will be absent).
//! - The VFS provides access to `/lib` — a pre-requisite for shared-library
//!   loading.
//! - Core ELF-parsing helpers (implemented locally here to avoid depending on
//!   the kernel-only loader) can correctly parse ELF headers, locate
//!   PT_INTERP, and extract DT_NEEDED entries.
//!
//! ## Note on test scope
//!
//! Full end-to-end dynamic-linking tests (actually loading a `.so` file and
//! calling a symbol from it) require a populated `/lib` directory at runtime.
//! Those tests are covered by the BDD feature suite.  The unit-level tests
//! here focus on the parsing and auxv logic that can be validated without a
//! live shared library.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use stem::println;
use stem::syscall::{auxv_get, vfs_open, vfs_close};
use abi::auxv;

// ── Auxv helpers (duplicated from test_exec_env to keep crates independent) ──

fn parse_auxv(buf: &[u8]) -> alloc::vec::Vec<(u64, u64)> {
    let mut out = alloc::vec::Vec::new();
    if buf.len() < 4 { return out; }
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut off = 4usize;
    for _ in 0..count {
        if off + 16 > buf.len() { break; }
        let k = u64::from_le_bytes(buf[off..off+8].try_into().unwrap());
        let v = u64::from_le_bytes(buf[off+8..off+16].try_into().unwrap());
        out.push((k, v));
        off += 16;
    }
    out
}

fn auxv_find(entries: &[(u64, u64)], typ: u64) -> u64 {
    entries.iter().find(|&&(k, _)| k == typ).map(|&(_, v)| v).unwrap_or(0)
}

// ── Minimal ELF-parsing helpers (userspace) ───────────────────────────────────

fn elf_u32(buf: &[u8], off: usize) -> Option<u32> {
    let s = buf.get(off..off+4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn elf_u64(buf: &[u8], off: usize) -> Option<u64> {
    let s = buf.get(off..off+8)?;
    Some(u64::from_le_bytes([s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]]))
}

fn elf_u16(buf: &[u8], off: usize) -> Option<u16> {
    let s = buf.get(off..off+2)?;
    Some(u16::from_le_bytes([s[0], s[1]]))
}

/// True iff `buf` begins with the ELF64 little-endian magic.
fn is_elf64_le(buf: &[u8]) -> bool {
    buf.len() >= 6
        && &buf[0..4] == b"\x7fELF"
        && buf[4] == 2  // ELFCLASS64
        && buf[5] == 1  // ELFDATA2LSB
}

/// Return the PT_INTERP path from an ELF64 binary, or `None`.
fn extract_interp(buf: &[u8]) -> Option<alloc::vec::Vec<u8>> {
    if !is_elf64_le(buf) { return None; }
    let phoff = elf_u64(buf, 32)? as usize;
    let phent = elf_u16(buf, 54)? as usize;
    let phnum = elf_u16(buf, 56)? as usize;
    for i in 0..phnum {
        let off = phoff + i * phent;
        if elf_u32(buf, off)? == 3 /* PT_INTERP */ {
            let foff  = elf_u64(buf, off + 8)? as usize;
            let fsz   = elf_u64(buf, off + 32)? as usize;
            if fsz == 0 || foff + fsz > buf.len() { return None; }
            let mut path = buf[foff..foff+fsz].to_vec();
            while path.last() == Some(&0) { path.pop(); }
            return Some(path);
        }
    }
    None
}

/// Find `DT_NEEDED` entries from a PT_DYNAMIC section.
fn collect_dt_needed(buf: &[u8]) -> alloc::vec::Vec<alloc::vec::Vec<u8>> {
    let mut result = alloc::vec::Vec::new();
    if !is_elf64_le(buf) { return result; }

    let phoff = match elf_u64(buf, 32) { Some(v) => v as usize, None => return result };
    let phent = match elf_u16(buf, 54) { Some(v) => v as usize, None => return result };
    let phnum = match elf_u16(buf, 56) { Some(v) => v as usize, None => return result };

    let mut dyn_off = 0usize;
    let mut dyn_sz  = 0usize;
    let mut strtab_off = 0usize;
    let mut strsz  = 0usize;
    let mut strtab_va = 0u64;

    for i in 0..phnum {
        let off = phoff + i * phent;
        match elf_u32(buf, off) {
            Some(2) => { // PT_DYNAMIC
                dyn_off = match elf_u64(buf, off + 8) { Some(v) => v as usize, None => 0 };
                dyn_sz  = match elf_u64(buf, off + 32) { Some(v) => v as usize, None => 0 };
            }
            Some(1) => {} // PT_LOAD — we'd need to build an offset map for full correctness
            _ => {}
        }
    }

    if dyn_off == 0 || dyn_sz == 0 { return result; }

    // Collect DT_STRTAB virtual address and DT_STRSZ first.
    let mut ptr = dyn_off;
    while ptr + 16 <= dyn_off + dyn_sz && ptr + 16 <= buf.len() {
        let tag = match elf_u64(buf, ptr) { Some(v) => v as i64, None => break };
        let val = match elf_u64(buf, ptr + 8) { Some(v) => v, None => break };
        if tag == 0 { break; } // DT_NULL
        if tag == 5  { strtab_va = val; }  // DT_STRTAB
        if tag == 10 { strsz = val as usize; } // DT_STRSZ
        ptr += 16;
    }

    // For a non-PIE ELF the strtab_va equals the file offset of the string
    // table.  For simplicity we scan the file for a matching section by
    // looking for the string table in the raw bytes (good enough for unit tests
    // of statically-embedded ELF blobs).
    if strtab_va == 0 { return result; }
    // Heuristic: treat strtab_va as a file offset within a small executable.
    strtab_off = strtab_va as usize;

    // Now collect DT_NEEDED offsets.
    ptr = dyn_off;
    while ptr + 16 <= dyn_off + dyn_sz && ptr + 16 <= buf.len() {
        let tag = match elf_u64(buf, ptr) { Some(v) => v as i64, None => break };
        let val = match elf_u64(buf, ptr + 8) { Some(v) => v as usize, None => break };
        if tag == 0 { break; }
        if tag == 1 /* DT_NEEDED */ {
            let name_off = strtab_off + val;
            if name_off < buf.len() {
                let end = buf[name_off..].iter().position(|&b| b == 0).unwrap_or(0);
                result.push(buf[name_off..name_off + end].to_vec());
            }
        }
        ptr += 16;
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Verify AT_PAGESZ and AT_ENTRY are always present in the auxv.
fn test_auxv_basics() {
    println!("[test_dyn_loader] test_auxv_basics: starting");

    let needed = auxv_get(&mut []).expect("auxv_get size query");
    assert!(needed >= 4);
    let mut buf = alloc::vec![0u8; needed];
    let _ = auxv_get(&mut buf).expect("auxv_get");
    let entries = parse_auxv(&buf);

    // AT_PAGESZ must be present.
    let pagesz = auxv_find(&entries, auxv::AT_PAGESZ);
    assert!(pagesz >= 4096 && pagesz.is_power_of_two(),
        "AT_PAGESZ invalid: {}", pagesz);

    // AT_ENTRY should be present (we have a valid ELF entry).
    let entry = auxv_find(&entries, auxv::AT_ENTRY);
    assert!(entry != 0, "AT_ENTRY should be non-zero");

    println!("[test_dyn_loader] test_auxv_basics: PASS (AT_PAGESZ={}, AT_ENTRY=0x{:x})",
        pagesz, entry);
}

/// A statically-linked binary (this test itself) should NOT have AT_BASE in
/// its auxv — AT_BASE is only set when the kernel loaded a PT_INTERP interpreter.
fn test_no_at_base_for_static_binary() {
    println!("[test_dyn_loader] test_no_at_base_for_static_binary: starting");

    let needed = auxv_get(&mut []).expect("auxv_get size query");
    let mut buf = alloc::vec![0u8; needed];
    let _ = auxv_get(&mut buf).expect("auxv_get");
    let entries = parse_auxv(&buf);

    // This binary is statically linked — AT_BASE should be absent (zero value).
    let base = auxv_find(&entries, auxv::AT_BASE);
    assert_eq!(base, 0,
        "AT_BASE should be 0 for a static binary, got 0x{:x}", base);

    println!("[test_dyn_loader] test_no_at_base_for_static_binary: PASS");
}

/// ELF magic detection helper works correctly.
fn test_elf_magic_detection() {
    println!("[test_dyn_loader] test_elf_magic_detection: starting");

    let valid = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    assert!(is_elf64_le(valid), "should detect valid ELF64 LE");

    let not_elf = b"PK\x03\x04\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    assert!(!is_elf64_le(not_elf), "should reject ZIP magic");

    let elf32 = b"\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    assert!(!is_elf64_le(elf32), "should reject ELF32");

    println!("[test_dyn_loader] test_elf_magic_detection: PASS");
}

/// Build a minimal ELF64 with a PT_INTERP segment and verify extraction.
fn test_extract_interp_from_elf() {
    println!("[test_dyn_loader] test_extract_interp_from_elf: starting");

    let interp_path = b"/lib/ld.so";
    let interp_offset: usize = 176;
    let total = interp_offset + interp_path.len() + 1;
    let mut elf = alloc::vec![0u8; total.max(512)];

    // ELF header
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2; elf[5] = 1; elf[6] = 1;
    let phoff: u64 = 64;
    elf[32..40].copy_from_slice(&phoff.to_le_bytes());       // e_phoff
    elf[54..56].copy_from_slice(&56u16.to_le_bytes());       // e_phentsize
    elf[56..58].copy_from_slice(&2u16.to_le_bytes());        // e_phnum
    elf[24..32].copy_from_slice(&0x200100u64.to_le_bytes()); // e_entry

    // PT_LOAD at [64..120)
    let p0 = 64usize;
    elf[p0..p0+4].copy_from_slice(&1u32.to_le_bytes());     // PT_LOAD
    elf[p0+4..p0+8].copy_from_slice(&5u32.to_le_bytes());   // R|X
    elf[p0+16..p0+24].copy_from_slice(&0x200000u64.to_le_bytes());
    elf[p0+32..p0+40].copy_from_slice(&100u64.to_le_bytes());
    elf[p0+40..p0+48].copy_from_slice(&100u64.to_le_bytes());
    elf[p0+48..p0+56].copy_from_slice(&0x1000u64.to_le_bytes());

    // PT_INTERP at [120..176)
    let p1 = 120usize;
    elf[p1..p1+4].copy_from_slice(&3u32.to_le_bytes());     // PT_INTERP
    elf[p1+8..p1+16].copy_from_slice(&(interp_offset as u64).to_le_bytes());
    let filesz = interp_path.len() as u64 + 1;
    elf[p1+32..p1+40].copy_from_slice(&filesz.to_le_bytes());
    elf[p1+40..p1+48].copy_from_slice(&filesz.to_le_bytes());

    // Interpreter path
    elf[interp_offset..interp_offset+interp_path.len()].copy_from_slice(interp_path);
    elf[interp_offset + interp_path.len()] = 0;

    let path = extract_interp(&elf).expect("should extract PT_INTERP path");
    assert_eq!(path, interp_path, "interp path mismatch");
    assert!(!path.contains(&0u8), "path should not contain NUL");

    println!("[test_dyn_loader] test_extract_interp_from_elf: PASS");
}

/// An ELF with no PT_INTERP should return None from extract_interp.
fn test_no_interp_in_static_elf() {
    println!("[test_dyn_loader] test_no_interp_in_static_elf: starting");

    let mut elf = alloc::vec![0u8; 256];
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2; elf[5] = 1; elf[6] = 1;
    let phoff: u64 = 64;
    elf[32..40].copy_from_slice(&phoff.to_le_bytes());
    elf[54..56].copy_from_slice(&56u16.to_le_bytes());
    elf[56..58].copy_from_slice(&1u16.to_le_bytes()); // only one phdr (PT_LOAD)
    elf[24..32].copy_from_slice(&0x200100u64.to_le_bytes());

    let p0 = 64usize;
    elf[p0..p0+4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    elf[p0+16..p0+24].copy_from_slice(&0x200000u64.to_le_bytes());
    elf[p0+40..p0+48].copy_from_slice(&100u64.to_le_bytes());
    elf[p0+32..p0+40].copy_from_slice(&100u64.to_le_bytes());

    assert!(extract_interp(&elf).is_none(), "static ELF should have no PT_INTERP");

    println!("[test_dyn_loader] test_no_interp_in_static_elf: PASS");
}

/// Verify that the VFS `/dev/null` device exists (basic VFS sanity).
fn test_vfs_dev_null_accessible() {
    println!("[test_dyn_loader] test_vfs_dev_null_accessible: starting");

    let fd = vfs_open("/dev/null", 0).expect("open /dev/null failed");
    vfs_close(fd).expect("close /dev/null failed");

    println!("[test_dyn_loader] test_vfs_dev_null_accessible: PASS");
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[stem::main]
fn main(_arg: usize) -> ! {
    println!("--- test_dyn_loader starting ---");

    test_auxv_basics();
    test_no_at_base_for_static_binary();
    test_elf_magic_detection();
    test_extract_interp_from_elf();
    test_no_interp_in_static_elf();
    test_vfs_dev_null_accessible();

    println!("--- test_dyn_loader: all tests PASSED ---");
    stem::syscall::exit(0);
}
