use crate::memory;
use crate::{
    BootModuleDesc, BootRuntime, BootTasking, FrameAllocatorHook, MapKind, MapPerms, UserEntry,
};
use abi::types::StackInfo;
use abi::vm::{VmBackingKind, VmMapFlags, VmProt, VmRegionInfo};
use core::cmp::{max, min};

struct LoaderAllocHook;
impl FrameAllocatorHook for LoaderAllocHook {
    fn alloc_frame(&self) -> Option<u64> {
        memory::alloc_frame()
    }
}

/// Information derived from the ELF image needed to build the auxiliary vector.
///
/// Populated by [`load_module`] when the image is a valid ELF64 binary; all
/// fields are zero for the flat binary fall-back path.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoaderAuxInfo {
    /// Virtual address (in the loaded address space) of the program-header table.
    pub phdr_vaddr: u64,
    /// Size of one ELF program-header entry (always 56 for ELF64).
    pub phent: u64,
    /// Number of program-header entries.
    pub phnum: u64,
    /// Actual entry-point virtual address (mirrors [`UserEntry::entry_pc`]).
    pub entry_vaddr: u64,

    // ── TLS (PT_TLS) information ────────────────────────────────────────────
    /// Biased virtual address of the TLS initialization template in the loaded
    /// image (PT_TLS p_vaddr + load_bias).  Zero when no PT_TLS segment is
    /// present.
    pub tls_template_vaddr: u64,
    /// Number of initialized bytes in the TLS template (PT_TLS p_filesz).
    pub tls_filesz: u64,
    /// Total per-thread TLS block size (PT_TLS p_memsz; includes BSS zeros).
    pub tls_memsz: u64,
    /// Required alignment for the per-thread TLS block (PT_TLS p_align).
    pub tls_align: u64,
    /// Initial thread pointer (FS_BASE on x86_64) pre-set by the kernel for the
    /// main / initial thread.  Zero when no PT_TLS segment is present.
    ///
    /// On x86_64 (ELF variant II), this is the address of the Thread Control
    /// Block (TCB) which contains a self-pointer at offset 0 and the per-thread
    /// TLS data at negative offsets (i.e. immediately before the TCB in memory).
    pub tls_tp: u64,

    // ── Dynamic linking (PT_INTERP) ─────────────────────────────────────────
    /// Base load address of the dynamic interpreter (AT_BASE).
    ///
    /// Non-zero only when the main executable has a `PT_INTERP` segment and the
    /// interpreter was successfully loaded into the address space by
    /// [`task_exec_current`].  The interpreter is mapped at this address.
    ///
    /// [`task_exec_current`]: crate::task::exec::task_exec_current
    pub interp_base: u64,
}

/// Load an ELF module or flat binary into `aspace` starting at `load_base`.
///
/// This is the implementation underlying both [`load_module`] (which uses the
/// default base `0x200000`) and the interpreter load path in exec.
pub fn load_module_at<R: BootRuntime>(
    rt: &R,
    aspace: <R::Tasking as BootTasking>::AddressSpace,
    module: &BootModuleDesc,
    load_base: u64,
) -> Option<(UserEntry, StackInfo, alloc::vec::Vec<VmRegionInfo>, LoaderAuxInfo)> {
    crate::kdebug!(
        "LOADER: Loading module '{}' (len={}, base=0x{:x})",
        module.name,
        module.bytes.len(),
        load_base,
    );
    if module.bytes.len() >= 16 {
        crate::ktrace!("  Header: {:02x?}", &module.bytes[0..16]);
    }

    let load_addr: u64 = load_base;
    let stack_top = 0x0080_0000;
    let reserve_bytes = 2 * 1024 * 1024;
    let guard_pages = 1usize;
    let initial_commit_bytes = 64 * 1024;
    let grow_chunk_bytes = 64 * 1024;

    let hook = LoaderAllocHook;
    let data_perms = MapPerms {
        user: true,
        read: true,
        write: true,
        exec: false,
        kind: MapKind::Normal,
    };

    let page_size = rt.page_size() as u64;
    let mut entry_pc = load_addr;
    let mut regions = alloc::vec::Vec::new();
    let mut aux_info = LoaderAuxInfo::default();

    // 1. Map ELF segments when available; otherwise fall back to a simple RWX layout.
    if let Some(mut elf) = parse_elf64(module.bytes) {
        // Sort segments by vaddr to ensure we process overlapping pages sequentially
        elf.load_segments.sort_by(|a, b| a.vaddr.cmp(&b.vaddr));

        let load_bias = load_addr.saturating_sub(elf.min_vaddr);
        entry_pc = elf.entry.saturating_add(load_bias);

        // Build auxv metadata from ELF header fields.
        aux_info = LoaderAuxInfo {
            phdr_vaddr: elf.phoff.saturating_add(load_bias),
            phent: elf.phentsize as u64,
            phnum: elf.phnum as u64,
            entry_vaddr: entry_pc,
            ..Default::default()
        };

        crate::kdebug!(
            "LOADER: ELF info: entry={:x}, min_vaddr={:x}, bias={:x}, entry_pc={:x}",
            elf.entry,
            elf.min_vaddr,
            load_bias,
            entry_pc
        );

        let mut last_virt_page = u64::MAX;
        let mut last_phys_page = 0;
        let mut last_perms = MapPerms {
            user: false,
            read: false,
            write: false,
            exec: false,
            kind: MapKind::Normal,
        };
        // Track the highest mapped virtual address end for TLS placement.
        let mut max_elf_vaddr_end: u64 = load_addr;

        for ph in elf.load_segments.iter() {
            let seg_vaddr = ph.vaddr.saturating_add(load_bias);
            let seg_mem_end = seg_vaddr.saturating_add(ph.memsz);
            if ph.memsz == 0 {
                continue;
            }

            let seg_start = align_down_u64(seg_vaddr, page_size);
            let seg_end = align_up_u64(seg_mem_end, page_size);
            // Track end for TLS placement.
            if seg_end > max_elf_vaddr_end {
                max_elf_vaddr_end = seg_end;
            }
            let mut perms = MapPerms {
                user: true,
                read: ph.read || ph.write || ph.exec,
                write: ph.write,
                exec: ph.exec,
                kind: MapKind::Normal,
            };
            crate::ktrace!("Segment: vaddr={:x} exec={}", seg_vaddr, perms.exec);

            // Record mapping
            let mut prot = VmProt::USER;
            if perms.read {
                prot |= VmProt::READ;
            }
            if perms.write {
                prot |= VmProt::WRITE;
            }
            if perms.exec {
                prot |= VmProt::EXEC;
            }

            regions.push(VmRegionInfo {
                start: seg_start as usize,
                end: seg_end as usize,
                prot,
                flags: VmMapFlags::empty(),
                backing_kind: VmBackingKind::Anonymous,
                _reserved: [0; 7],
            });

            let mut virt = seg_start;
            while virt < seg_end {
                let phys;
                let mut reuse_page = false;
                let mut page_perms = perms;

                if virt == last_virt_page {
                    // Overlap detected! Reuse the previous page and merge permissions.
                    phys = last_phys_page;
                    reuse_page = true;
                    page_perms = match merge_perms(last_perms, perms) {
                        Ok(p) => p,
                        Err(e) => {
                            crate::kerror!("ERROR: {} at {:x}", e, virt);
                            return None;
                        }
                    };
                    crate::ktrace!(
                        "  Overlap at {:x}: merging perms to r={} w={} x={}",
                        virt,
                        page_perms.read,
                        page_perms.write,
                        page_perms.exec
                    );
                } else {
                    // New page
                    phys = memory::alloc_frame().expect("OOM loading module segment");
                }

                let hhdm_virt = phys + rt.phys_to_virt_offset();

                if !reuse_page {
                    unsafe {
                        core::ptr::write_bytes(hhdm_virt as *mut u8, 0, page_size as usize);
                    }
                }

                let page_end = virt.saturating_add(page_size);
                let file_start = seg_vaddr;
                let file_end = seg_vaddr.saturating_add(ph.filesz);
                let copy_start = max(virt, file_start);
                let copy_end = min(page_end, file_end);

                if copy_start < copy_end {
                    let src_off = ph.offset.saturating_add(copy_start - seg_vaddr);
                    let len = (copy_end - copy_start) as usize;
                    let dst = (hhdm_virt + (copy_start - virt)) as *mut u8;
                    if src_off as usize + len <= module.bytes.len() {
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                module.bytes.as_ptr().add(src_off as usize),
                                dst,
                                len,
                            );
                        }

                        if copy_start <= 0x201420 && 0x201420 < copy_end {
                            let off_in_page = (0x201420 - copy_start) as usize;
                            unsafe {
                                let bytes = core::slice::from_raw_parts(dst.add(off_in_page), 8);
                                crate::ktrace!("  COPIED at 0x201420: {:02x?}", bytes);
                            }
                        }
                    } else {
                        return None;
                    }
                }

                rt.tasking()
                    .map_page(aspace, virt, phys, page_perms, MapKind::Normal, &hook)
                    .unwrap();

                last_virt_page = virt;
                last_phys_page = phys;
                last_perms = page_perms;

                virt += page_size;
            }
        }

        // 2. Allocate and initialize the per-thread TLS block for the main thread,
        //    if the ELF image has a PT_TLS segment.
        if let Some(tls) = &elf.tls {
            let tls_tp = setup_initial_tls_block(
                rt,
                aspace,
                tls,
                module.bytes,
                max_elf_vaddr_end,
                page_size,
                data_perms,
                &hook,
                &mut regions,
                load_bias,
            );
            if let Some(tp) = tls_tp {
                aux_info.tls_template_vaddr = tls.vaddr.saturating_add(load_bias);
                aux_info.tls_filesz = tls.filesz;
                aux_info.tls_memsz = tls.memsz;
                aux_info.tls_align = tls.align;
                aux_info.tls_tp = tp;
                crate::kdebug!(
                    "LOADER: TLS block: tp={:#x} filesz={} memsz={} align={}",
                    tp,
                    tls.filesz,
                    tls.memsz,
                    tls.align
                );
            } else {
                crate::kerror!("LOADER: Failed to set up TLS block for module '{}'", module.name);
                return None;
            }
        }
    } else {
        // Hardcoded load address for simple PIE/or-not-PIE loading.
        // Fixed address 0x200000 is fine for the main executable today.
        let text_perms = MapPerms {
            user: true,
            read: true,
            write: true,
            exec: true,
            kind: MapKind::Normal,
        };
        let mut virt = load_addr as u64;

        // Record mapping
        let prot = VmProt::USER | VmProt::READ | VmProt::WRITE | VmProt::EXEC;
        let len = align_up_u64(module.bytes.len() as u64, page_size);
        regions.push(VmRegionInfo {
            start: load_addr as usize,
            end: (load_addr + len) as usize,
            prot,
            flags: VmMapFlags::empty(),
            backing_kind: VmBackingKind::Anonymous,
            _reserved: [0; 7],
        });

        if page_size == 0 {
            panic!("LOADER: page_size is zero!");
        }

        for chunk in module.bytes.chunks(page_size as usize) {
            let phys = memory::alloc_frame().expect("OOM loading module");
            let hhdm_virt = phys + rt.phys_to_virt_offset();
            unsafe {
                core::ptr::copy_nonoverlapping(chunk.as_ptr(), hhdm_virt as *mut u8, chunk.len());
                if chunk.len() < page_size as usize {
                    core::ptr::write_bytes(
                        (hhdm_virt as *mut u8).add(chunk.len()),
                        0,
                        page_size as usize - chunk.len(),
                    );
                }
            }

            rt.tasking()
                .map_page(aspace, virt, phys, text_perms, MapKind::Normal, &hook)
                .unwrap();
            virt += page_size;
        }
    }

    // 3. Map Stack (guard + reserve with initial commit)
    let guard_bytes = (guard_pages as u64).saturating_mul(page_size);
    let reserve_bytes = align_up_u64(reserve_bytes as u64, page_size);
    let total = guard_bytes.saturating_add(reserve_bytes);
    let reserve_end = stack_top as u64;
    let base = reserve_end.saturating_sub(total);
    let guard_start = base;
    let guard_end = base.saturating_add(guard_bytes);
    let reserve_start = guard_end;
    let commit_len = align_up_u64(initial_commit_bytes as u64, page_size);
    let commit_start = reserve_end.saturating_sub(commit_len);

    // Record stack mapping (committed part)
    regions.push(VmRegionInfo {
        start: commit_start as usize,
        end: reserve_end as usize,
        prot: VmProt::USER | VmProt::READ | VmProt::WRITE,
        flags: VmMapFlags::empty(),
        backing_kind: VmBackingKind::Anonymous,
        _reserved: [0; 7],
    });

    let mut virt = commit_start;
    while virt < reserve_end {
        let phys = memory::alloc_frame().expect("OOM loading stack");
        let hhdm_virt = phys + rt.phys_to_virt_offset();
        unsafe {
            core::ptr::write_bytes(hhdm_virt as *mut u8, 0, page_size as usize);
        }
        rt.tasking()
            .map_page(aspace, virt, phys, data_perms, MapKind::Normal, &hook)
            .unwrap();
        virt += page_size;
    }

    // Ensure instruction cache sees freshly loaded code
    rt.icache_invalidate();

    let stack_info = StackInfo {
        guard_start: guard_start as usize,
        guard_end: guard_end as usize,
        reserve_start: reserve_start as usize,
        reserve_end: reserve_end as usize,
        committed_start: commit_start as usize,
        grow_chunk_bytes,
    };

    if entry_pc == 0 {
        crate::kerror!("LOADER: Invalid entry point 0 for module '{}'", module.name);
        return None;
    }

    Some((
        UserEntry {
            entry_pc: entry_pc as usize,
            user_sp: stack_top,
            arg0: 0,
        },
        stack_info,
        regions,
        aux_info,
    ))
}

/// Load an ELF module or flat binary into `aspace` at the default base address
/// (`0x200000`).  This is the standard entry point used by boot-time loaders
/// and `spawn_process*`.
pub fn load_module<R: BootRuntime>(
    rt: &R,
    aspace: <R::Tasking as BootTasking>::AddressSpace,
    module: &BootModuleDesc,
) -> Option<(UserEntry, StackInfo, alloc::vec::Vec<VmRegionInfo>, LoaderAuxInfo)> {
    load_module_at::<R>(rt, aspace, module, 0x200000)
}

/// Extract the dynamic interpreter path from a PT_INTERP segment, if present.
///
/// Returns `None` when the binary does not have a PT_INTERP segment or when
/// the ELF cannot be parsed.  The returned slice is a null-terminated path
/// string copied from the file (trailing NUL stripped).
pub fn extract_interp_path(bytes: &[u8]) -> Option<alloc::vec::Vec<u8>> {
    if bytes.len() < 64 {
        return None;
    }
    if &bytes[0..4] != b"\x7fELF" {
        return None;
    }
    if bytes[4] != 2 || bytes[5] != 1 {
        return None; // not ELF64 LE
    }
    let e_phoff = read_u64(bytes, 32)?;
    let e_phentsize = read_u16(bytes, 54)? as u64;
    let e_phnum = read_u16(bytes, 56)? as u64;
    if e_phoff == 0 || e_phentsize == 0 || e_phnum == 0 {
        return None;
    }
    for i in 0..e_phnum {
        let off = e_phoff.saturating_add(i.saturating_mul(e_phentsize)) as usize;
        let p_type = read_u32(bytes, off)?;
        if p_type == 3 {
            // PT_INTERP
            let p_offset = read_u64(bytes, off + 8)? as usize;
            let p_filesz = read_u64(bytes, off + 32)? as usize;
            if p_filesz == 0 || p_offset + p_filesz > bytes.len() {
                return None;
            }
            let mut path = bytes[p_offset..p_offset + p_filesz].to_vec();
            // Strip trailing NUL bytes.
            while path.last() == Some(&0) {
                path.pop();
            }
            return Some(path);
        }
    }
    None
}

fn align_up_u64(value: u64, align: u64) -> u64 {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

fn align_down_u64(value: u64, align: u64) -> u64 {
    if align == 0 {
        return value;
    }
    value & !(align - 1)
}

/// Allocate and initialize the per-thread TLS block for the initial (main) thread.
///
/// Layout (ELF x86_64 "variant II"):
/// ```text
/// [ TLS data (memsz bytes, aligned to tls.align) ][ TCB (16 bytes) ]
///  ^tls_block_vaddr                                ^TP (= tls_tp)
/// ```
/// - Bytes `[0, filesz)` are copied from the ELF file's TLS template.
/// - Bytes `[filesz, memsz_aligned)` are zeroed (BSS).
/// - The TCB starts at offset `memsz_aligned` from `tls_block_vaddr`.
/// - `TCB[0]` (a `u64`) is set to `TP` (self-pointer required by x86_64 ABI).
/// - `TP` = `tls_block_vaddr + memsz_aligned` is returned on success.
///
/// Returns `None` on allocation failure or if the TLS template is inconsistent.
#[allow(clippy::too_many_arguments)]
fn setup_initial_tls_block<R: BootRuntime>(
    rt: &R,
    aspace: <R::Tasking as BootTasking>::AddressSpace,
    tls: &ElfTlsSegment,
    elf_bytes: &[u8],
    after_vaddr: u64,
    page_size: u64,
    perms: MapPerms,
    hook: &LoaderAllocHook,
    regions: &mut alloc::vec::Vec<VmRegionInfo>,
    _load_bias: u64,
) -> Option<u64> {
    // ── Geometry ───────────────────────────────────────────────────────────
    let tls_align = tls.align.max(16);
    // Round memsz up to the required TLS alignment so the TCB starts aligned.
    let tls_data_size = align_up_u64(tls.memsz, tls_align);
    // The TCB needs at least 16 bytes: [self_ptr (u64), dtv_ptr (u64)].
    let tcb_size: u64 = 16;
    let total_size = tls_data_size.saturating_add(tcb_size);

    // ── Choose a virtual address ────────────────────────────────────────────
    // Place the TLS block right after the highest ELF segment, page-aligned.
    let tls_block_vaddr = align_up_u64(after_vaddr, page_size.max(tls_align));
    let tls_block_end = tls_block_vaddr.saturating_add(align_up_u64(total_size, page_size));

    // ── Allocate pages ─────────────────────────────────────────────────────
    let mut page_hhdms: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
    let mut virt = tls_block_vaddr;
    while virt < tls_block_end {
        let phys = memory::alloc_frame()?;
        let hhdm = phys + rt.phys_to_virt_offset();
        // Zero the whole page.
        unsafe {
            core::ptr::write_bytes(hhdm as *mut u8, 0, page_size as usize);
        }
        rt.tasking()
            .map_page(aspace, virt, phys, perms, MapKind::Normal, hook)
            .ok()?;
        page_hhdms.push(hhdm);
        virt += page_size;
    }

    // ── Copy TLS initialization template ───────────────────────────────────
    if tls.filesz > 0 {
        let file_end = (tls.offset as usize).saturating_add(tls.filesz as usize);
        if file_end > elf_bytes.len() {
            return None;
        }
        let template = &elf_bytes[tls.offset as usize..file_end];
        // Write template bytes at the start of the TLS data area.
        hhdm_write_bytes(&page_hhdms, 0, template, page_size);
    }
    // (BSS bytes are already zeroed from the page allocation above.)

    // ── Set up the TCB self-pointer ─────────────────────────────────────────
    // TP = start of TCB = tls_block_vaddr + tls_data_size.
    let tp: u64 = tls_block_vaddr.saturating_add(tls_data_size);
    // Write the self-pointer: *((u64*)tp) = tp.
    hhdm_write_u64(&page_hhdms, tls_data_size, tp, page_size);

    // ── Record VM region ───────────────────────────────────────────────────
    regions.push(VmRegionInfo {
        start: tls_block_vaddr as usize,
        end: tls_block_end as usize,
        prot: VmProt::USER | VmProt::READ | VmProt::WRITE,
        flags: VmMapFlags::empty(),
        backing_kind: VmBackingKind::Anonymous,
        _reserved: [0; 7],
    });

    Some(tp)
}

/// Write `data` bytes into a sequence of HHDM-mapped pages starting at `block_offset`
/// bytes from the beginning of the first page.
fn hhdm_write_bytes(page_hhdms: &[u64], block_offset: u64, data: &[u8], page_size: u64) {
    if data.is_empty() || page_hhdms.is_empty() {
        return;
    }
    // Optimised: work page by page rather than byte by byte.
    let mut remaining = data;
    let mut offset = block_offset;
    while !remaining.is_empty() {
        let page_idx = (offset / page_size) as usize;
        if page_idx >= page_hhdms.len() {
            break;
        }
        let page_off = (offset % page_size) as usize;
        let space_in_page = (page_size as usize).saturating_sub(page_off);
        let copy_len = space_in_page.min(remaining.len());
        let dst = (page_hhdms[page_idx] + page_off as u64) as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(remaining.as_ptr(), dst, copy_len);
        }
        remaining = &remaining[copy_len..];
        offset += copy_len as u64;
    }
}

/// Write a single `u64` value at `block_offset` bytes from the start of the
/// first TLS block page, in little-endian byte order.
fn hhdm_write_u64(page_hhdms: &[u64], block_offset: u64, value: u64, page_size: u64) {
    hhdm_write_bytes(page_hhdms, block_offset, &value.to_le_bytes(), page_size);
}

struct ElfLoadSegment {
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    read: bool,
    write: bool,
    exec: bool,
}

/// Information from the ELF PT_TLS program header describing the per-thread
/// TLS template and block geometry.
struct ElfTlsSegment {
    /// File offset of the TLS initialization image.
    offset: u64,
    /// Virtual address of the TLS template in the ELF image.
    vaddr: u64,
    /// Initialized bytes (copy from file into each thread's TLS area).
    filesz: u64,
    /// Total per-thread TLS block size (filesz + BSS zeros).
    memsz: u64,
    /// Required alignment for the TLS data block.
    align: u64,
}

struct ElfInfo {
    entry: u64,
    min_vaddr: u64,
    /// File offset of the program-header table (used for AT_PHDR via load bias).
    phoff: u64,
    /// Size of one program-header entry in bytes (AT_PHENT).
    phentsize: u16,
    /// Number of program-header entries (AT_PHNUM).
    phnum: u64,
    load_segments: alloc::vec::Vec<ElfLoadSegment>,
    /// TLS segment, if a PT_TLS program header was found.
    tls: Option<ElfTlsSegment>,
}

fn parse_elf64(bytes: &[u8]) -> Option<ElfInfo> {
    if bytes.len() < 64 {
        return None;
    }
    if &bytes[0..4] != b"\x7fELF" {
        return None;
    }
    if bytes[4] != 2 || bytes[5] != 1 {
        return None;
    }
    let e_entry = read_u64(bytes, 24)?;
    let e_phoff = read_u64(bytes, 32)?;
    let e_phentsize = read_u16(bytes, 54)?;
    let e_phnum = read_u16(bytes, 56)? as u64;
    if e_phoff == 0 || e_phentsize == 0 || e_phnum == 0 {
        return None;
    }

    let mut min_vaddr = u64::MAX;
    let mut load_segments = alloc::vec::Vec::new();
    let mut tls: Option<ElfTlsSegment> = None;
    for i in 0..e_phnum {
        let off = e_phoff.saturating_add(i.saturating_mul(e_phentsize as u64)) as usize;
        let p_type = read_u32(bytes, off)?;
        if p_type == 1 {
            // PT_LOAD
            let p_flags = read_u32(bytes, off + 4)?;
            let p_offset = read_u64(bytes, off + 8)?;
            let p_vaddr = read_u64(bytes, off + 16)?;
            let p_filesz = read_u64(bytes, off + 32)?;
            let p_memsz = read_u64(bytes, off + 40)?;

            min_vaddr = min(min_vaddr, p_vaddr);
            load_segments.push(ElfLoadSegment {
                offset: p_offset,
                vaddr: p_vaddr,
                filesz: p_filesz,
                memsz: p_memsz,
                read: (p_flags & 0x4) != 0,
                write: (p_flags & 0x2) != 0,
                exec: (p_flags & 0x1) != 0,
            });
        } else if p_type == 7 {
            // PT_TLS — describes the per-thread TLS template.
            let p_offset = read_u64(bytes, off + 8)?;
            let p_vaddr = read_u64(bytes, off + 16)?;
            let p_filesz = read_u64(bytes, off + 32)?;
            let p_memsz = read_u64(bytes, off + 40)?;
            let p_align = read_u64(bytes, off + 48).unwrap_or(1);
            tls = Some(ElfTlsSegment {
                offset: p_offset,
                vaddr: p_vaddr,
                filesz: p_filesz,
                memsz: p_memsz,
                align: if p_align == 0 { 1 } else { p_align },
            });
        }
    }

    if load_segments.is_empty() || min_vaddr == u64::MAX {
        return None;
    }

    Some(ElfInfo {
        entry: e_entry,
        min_vaddr,
        phoff: e_phoff,
        phentsize: e_phentsize,
        phnum: e_phnum,
        load_segments,
        tls,
    })
}

fn read_u16(bytes: &[u8], off: usize) -> Option<u16> {
    let slice = bytes.get(off..off + 2)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], off: usize) -> Option<u32> {
    let slice = bytes.get(off..off + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], off: usize) -> Option<u64> {
    let slice = bytes.get(off..off + 8)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn merge_perms(last: MapPerms, next: MapPerms) -> Result<MapPerms, &'static str> {
    let merged = MapPerms {
        user: last.user || next.user,
        read: last.read || next.read,
        write: last.write || next.write,
        exec: last.exec || next.exec,
        kind: last.kind,
    };

    // Enforce W^X: never produce RWX
    if merged.write && merged.exec {
        return Err("Permission conflict: merging results in RWX (W^X violation)");
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_perms() {
        let r = MapPerms {
            user: true,
            read: true,
            write: false,
            exec: false,
            kind: MapKind::Normal,
        };
        let rw = MapPerms {
            user: true,
            read: true,
            write: true,
            exec: false,
            kind: MapKind::Normal,
        };
        let rx = MapPerms {
            user: true,
            read: true,
            write: false,
            exec: true,
            kind: MapKind::Normal,
        };
        let x = MapPerms {
            user: true,
            read: false,
            write: false,
            exec: true,
            kind: MapKind::Normal,
        };

        // RX + RW -> Error
        assert!(merge_perms(rx, rw).is_err());

        // RX + R -> RX
        let res = merge_perms(rx, r).expect("RX + R failed");
        assert!(res.read && !res.write && res.exec);

        // RW + R -> RW
        let res = merge_perms(rw, r).expect("RW + R failed");
        assert!(res.read && res.write && !res.exec);

        // RX + X -> RX (normalize)
        let res = merge_perms(rx, x).expect("RX + X failed");
        assert!(res.read && !res.write && res.exec);

        // R + R -> R
        let res = merge_perms(r, r).expect("R + R failed");
        assert!(res.read && !res.write && !res.exec);
    }

    /// Build a minimal ELF64 binary containing one PT_LOAD and one PT_TLS
    /// program header so we can unit-test `parse_elf64` TLS extraction.
    fn build_elf64_with_tls(
        entry: u64,
        load_vaddr: u64,
        load_filesz: u64,
        load_memsz: u64,
        tls_offset: u64,
        tls_vaddr: u64,
        tls_filesz: u64,
        tls_memsz: u64,
        tls_align: u64,
    ) -> alloc::vec::Vec<u8> {
        let mut bytes = alloc::vec![0u8; 512];

        // ELF magic + class/data/version
        bytes[0..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2; // EI_CLASS: ELFCLASS64
        bytes[5] = 1; // EI_DATA:  ELFDATA2LSB (little-endian)
        bytes[6] = 1; // EI_VERSION

        let phoff: u64 = 64; // program headers start immediately after ELF header
        let e_phentsize: u16 = 56;
        let e_phnum: u16 = 2; // PT_LOAD + PT_TLS

        bytes[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry
        bytes[32..40].copy_from_slice(&phoff.to_le_bytes()); // e_phoff
        bytes[54..56].copy_from_slice(&e_phentsize.to_le_bytes()); // e_phentsize
        bytes[56..58].copy_from_slice(&e_phnum.to_le_bytes()); // e_phnum

        // PT_LOAD at file offset 64
        let p0 = 64usize;
        bytes[p0..p0 + 4].copy_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
        bytes[p0 + 4..p0 + 8].copy_from_slice(&5u32.to_le_bytes()); // p_flags = R|X
        bytes[p0 + 8..p0 + 16].copy_from_slice(&0u64.to_le_bytes()); // p_offset
        bytes[p0 + 16..p0 + 24].copy_from_slice(&load_vaddr.to_le_bytes()); // p_vaddr
        bytes[p0 + 24..p0 + 32].copy_from_slice(&load_vaddr.to_le_bytes()); // p_paddr
        bytes[p0 + 32..p0 + 40].copy_from_slice(&load_filesz.to_le_bytes()); // p_filesz
        bytes[p0 + 40..p0 + 48].copy_from_slice(&load_memsz.to_le_bytes()); // p_memsz
        bytes[p0 + 48..p0 + 56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

        // PT_TLS at file offset 64 + 56 = 120
        let p1 = p0 + 56;
        bytes[p1..p1 + 4].copy_from_slice(&7u32.to_le_bytes()); // p_type = PT_TLS
        bytes[p1 + 4..p1 + 8].copy_from_slice(&4u32.to_le_bytes()); // p_flags = R
        bytes[p1 + 8..p1 + 16].copy_from_slice(&tls_offset.to_le_bytes()); // p_offset
        bytes[p1 + 16..p1 + 24].copy_from_slice(&tls_vaddr.to_le_bytes()); // p_vaddr
        bytes[p1 + 24..p1 + 32].copy_from_slice(&tls_vaddr.to_le_bytes()); // p_paddr
        bytes[p1 + 32..p1 + 40].copy_from_slice(&tls_filesz.to_le_bytes()); // p_filesz
        bytes[p1 + 40..p1 + 48].copy_from_slice(&tls_memsz.to_le_bytes()); // p_memsz
        bytes[p1 + 48..p1 + 56].copy_from_slice(&tls_align.to_le_bytes()); // p_align

        bytes
    }

    #[test]
    fn test_parse_elf64_extracts_tls_segment() {
        let bytes = build_elf64_with_tls(
            0x200100, // entry
            0x200000, // PT_LOAD vaddr
            200,      // PT_LOAD filesz
            200,      // PT_LOAD memsz
            256,      // PT_TLS file offset
            0x201000, // PT_TLS vaddr
            64,       // PT_TLS filesz
            128,      // PT_TLS memsz
            16,       // PT_TLS align
        );

        let info = parse_elf64(&bytes).expect("parse_elf64 should succeed");

        // PT_LOAD must be found.
        assert_eq!(info.load_segments.len(), 1, "expected one PT_LOAD segment");
        assert_eq!(info.min_vaddr, 0x200000);
        assert_eq!(info.entry, 0x200100);

        // PT_TLS must be parsed correctly.
        let tls = info.tls.expect("PT_TLS segment should be extracted");
        assert_eq!(tls.offset, 256, "p_offset");
        assert_eq!(tls.vaddr, 0x201000, "p_vaddr");
        assert_eq!(tls.filesz, 64, "p_filesz");
        assert_eq!(tls.memsz, 128, "p_memsz");
        assert_eq!(tls.align, 16, "p_align");
    }

    #[test]
    fn test_parse_elf64_no_tls_when_absent() {
        // Build an ELF with only a PT_LOAD, no PT_TLS.
        let mut bytes = alloc::vec![0u8; 256];
        bytes[0..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        let phoff: u64 = 64;
        let e_phentsize: u16 = 56;
        let e_phnum: u16 = 1; // only PT_LOAD
        bytes[24..32].copy_from_slice(&0x200100u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&phoff.to_le_bytes());
        bytes[54..56].copy_from_slice(&e_phentsize.to_le_bytes());
        bytes[56..58].copy_from_slice(&e_phnum.to_le_bytes());

        let p0 = 64usize;
        bytes[p0..p0 + 4].copy_from_slice(&1u32.to_le_bytes());
        bytes[p0 + 16..p0 + 24].copy_from_slice(&0x200000u64.to_le_bytes());
        bytes[p0 + 40..p0 + 48].copy_from_slice(&100u64.to_le_bytes());
        bytes[p0 + 32..p0 + 40].copy_from_slice(&100u64.to_le_bytes());

        let info = parse_elf64(&bytes).expect("should parse without TLS");
        assert!(info.tls.is_none(), "tls should be None when no PT_TLS");
        assert_eq!(info.load_segments.len(), 1);
    }

    #[test]
    fn test_parse_elf64_tls_align_zero_becomes_one() {
        // A PT_TLS with p_align = 0 should be normalized to 1.
        let bytes = build_elf64_with_tls(
            0x200100, 0x200000, 200, 200, 256, 0x201000, 64, 128,
            0, // p_align = 0 → should become 1
        );
        let info = parse_elf64(&bytes).expect("parse should succeed");
        let tls = info.tls.expect("TLS should be present");
        assert_eq!(tls.align, 1, "align=0 should be normalized to 1");
    }

    #[test]
    fn test_hhdm_write_bytes_across_page_boundary() {
        // Verify that hhdm_write_bytes correctly spans multiple pages.
        let page_size: u64 = 64; // small page size for test purposes
        let mut page0 = alloc::vec![0u8; page_size as usize];
        let mut page1 = alloc::vec![0u8; page_size as usize];
        let hhdms = alloc::vec![page0.as_mut_ptr() as u64, page1.as_mut_ptr() as u64];

        // Write 16 bytes starting at offset 56 (8 bytes in page0, 8 bytes in page1).
        let data = [0xABu8; 16];
        hhdm_write_bytes(&hhdms, 56, &data, page_size);

        // Last 8 bytes of page0 should be 0xAB.
        assert!(page0[56..64].iter().all(|&b| b == 0xAB));
        // First 8 bytes of page1 should be 0xAB.
        assert!(page1[0..8].iter().all(|&b| b == 0xAB));
        // Earlier bytes in page0 should still be zero.
        assert!(page0[0..56].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_hhdm_write_u64_self_pointer() {
        // Simulates writing the TCB self-pointer.
        let page_size: u64 = 4096;
        let mut page = alloc::vec![0u8; page_size as usize];
        let hhdms = alloc::vec![page.as_mut_ptr() as u64];
        let tp: u64 = 0xDEAD_BEEF_1234_5678;
        hhdm_write_u64(&hhdms, 0, tp, page_size);
        let written = u64::from_le_bytes(page[0..8].try_into().unwrap());
        assert_eq!(written, tp);
    }

    // ── extract_interp_path tests ─────────────────────────────────────────────

    /// Build a minimal ELF64 with a PT_INTERP segment containing `interp`.
    fn build_elf64_with_interp(interp: &[u8]) -> alloc::vec::Vec<u8> {
        // Layout:
        //   [0..64)   ELF header
        //   [64..120) PT_LOAD program header (56 bytes)
        //   [120..176) PT_INTERP program header (56 bytes)
        //   [176..176+interp.len()+1) interpreter path (NUL-terminated)
        let interp_offset: usize = 176;
        let total = interp_offset + interp.len() + 1; // +1 for NUL
        let mut bytes = alloc::vec![0u8; total.max(512)];

        bytes[0..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2; // ELFCLASS64
        bytes[5] = 1; // ELFDATA2LSB
        bytes[6] = 1; // EV_CURRENT

        let e_phoff: u64 = 64;
        let e_phentsize: u16 = 56;
        let e_phnum: u16 = 2; // PT_LOAD + PT_INTERP

        bytes[24..32].copy_from_slice(&0x200100u64.to_le_bytes()); // e_entry
        bytes[32..40].copy_from_slice(&e_phoff.to_le_bytes());     // e_phoff
        bytes[54..56].copy_from_slice(&e_phentsize.to_le_bytes()); // e_phentsize
        bytes[56..58].copy_from_slice(&e_phnum.to_le_bytes());     // e_phnum

        // PT_LOAD at [64..120)
        let p0 = 64usize;
        bytes[p0..p0 + 4].copy_from_slice(&1u32.to_le_bytes()); // p_type=PT_LOAD
        bytes[p0 + 4..p0 + 8].copy_from_slice(&5u32.to_le_bytes()); // p_flags=R|X
        bytes[p0 + 16..p0 + 24].copy_from_slice(&0x200000u64.to_le_bytes()); // p_vaddr
        bytes[p0 + 32..p0 + 40].copy_from_slice(&100u64.to_le_bytes()); // p_filesz
        bytes[p0 + 40..p0 + 48].copy_from_slice(&100u64.to_le_bytes()); // p_memsz
        bytes[p0 + 48..p0 + 56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

        // PT_INTERP at [120..176)
        let p1 = 120usize;
        bytes[p1..p1 + 4].copy_from_slice(&3u32.to_le_bytes()); // p_type=PT_INTERP
        bytes[p1 + 8..p1 + 16].copy_from_slice(&(interp_offset as u64).to_le_bytes()); // p_offset
        let filesz = interp.len() as u64 + 1; // include NUL
        bytes[p1 + 32..p1 + 40].copy_from_slice(&filesz.to_le_bytes()); // p_filesz
        bytes[p1 + 40..p1 + 48].copy_from_slice(&filesz.to_le_bytes()); // p_memsz

        // Interpreter path (NUL-terminated)
        bytes[interp_offset..interp_offset + interp.len()].copy_from_slice(interp);
        bytes[interp_offset + interp.len()] = 0;

        bytes
    }

    #[test]
    fn test_extract_interp_path_present() {
        let interp = b"/lib/ld.so.1";
        let bytes = build_elf64_with_interp(interp);
        let path = extract_interp_path(&bytes).expect("should extract interp path");
        assert_eq!(path, interp, "interp path mismatch");
    }

    #[test]
    fn test_extract_interp_path_absent() {
        // An ELF with only PT_LOAD — no PT_INTERP.
        let bytes = build_elf64_with_tls(
            0x200100, 0x200000, 100, 100, 256, 0x201000, 8, 16, 8,
        );
        assert!(
            extract_interp_path(&bytes).is_none(),
            "should return None when PT_INTERP absent"
        );
    }

    #[test]
    fn test_extract_interp_path_nul_stripped() {
        // Ensure trailing NUL bytes are stripped from the returned path.
        let interp = b"/lib/ld-thingos.so";
        let bytes = build_elf64_with_interp(interp);
        let path = extract_interp_path(&bytes).expect("should extract");
        assert!(
            !path.contains(&0u8),
            "returned path should not contain NUL bytes"
        );
        assert_eq!(path, interp);
    }

    #[test]
    fn test_extract_interp_path_invalid_elf() {
        let garbage = alloc::vec![0xFFu8; 512];
        assert!(extract_interp_path(&garbage).is_none());
    }
}
