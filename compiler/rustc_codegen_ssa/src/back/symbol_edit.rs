//! Binary-level symbol editing for staticlib post-processing.
//!
//! - **Hide**: sets STV_HIDDEN (ELF) or N_PEXT (Mach-O) on non-exported symbols.
//! - **Rename**: appends a vendor-specific suffix to non-exported symbol names by
//!   rebuilding the string table.

use std::borrow::Cow;
use std::mem;

use object::read::elf::{SectionHeader as _, Sym as _};
use object::read::macho::Nlist;
use object::{Endianness, elf, macho};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

struct Patch {
    offset: usize,
    value: u8,
}

struct RenameEntry {
    name_field_offset: usize,
    name: String,
}

pub(super) fn apply_edits<'a>(
    data: &'a [u8],
    exported: &FxHashSet<String>,
    hide: bool,
    rename: Option<&(FxHashSet<String>, &str)>,
) -> Cow<'a, [u8]> {
    let result = match object::File::parse(data).ok() {
        Some(object::File::Elf64(_)) => elf_edit_impl::<elf::FileHeader64<Endianness>>(
            data,
            exported,
            hide,
            rename,
            mem::offset_of!(elf::Sym64<Endianness>, st_other),
        ),
        Some(object::File::Elf32(_)) => elf_edit_impl::<elf::FileHeader32<Endianness>>(
            data,
            exported,
            hide,
            rename,
            mem::offset_of!(elf::Sym32<Endianness>, st_other),
        ),
        Some(object::File::MachO64(_)) => macho_edit_impl::<macho::MachHeader64<Endianness>>(
            data,
            exported,
            hide,
            rename,
            mem::offset_of!(macho::Nlist64<Endianness>, n_type),
        ),
        Some(object::File::MachO32(_)) => macho_edit_impl::<macho::MachHeader32<Endianness>>(
            data,
            exported,
            hide,
            rename,
            mem::offset_of!(macho::Nlist32<Endianness>, n_type),
        ),
        _ => None,
    };
    match result {
        Some(v) => Cow::Owned(v),
        None => Cow::Borrowed(data),
    }
}

pub(super) fn collect_internal_names(
    data: &[u8],
    exported: &FxHashSet<String>,
    out: &mut FxHashSet<String>,
) {
    let Ok(file) = object::File::parse(data) else { return };
    match file {
        object::File::Elf64(_) => {
            elf_collect_impl::<elf::FileHeader64<Endianness>>(data, exported, out)
        }
        object::File::Elf32(_) => {
            elf_collect_impl::<elf::FileHeader32<Endianness>>(data, exported, out)
        }
        object::File::MachO64(_) => {
            macho_collect_impl::<macho::MachHeader64<Endianness>>(data, exported, out)
        }
        object::File::MachO32(_) => {
            macho_collect_impl::<macho::MachHeader32<Endianness>>(data, exported, out)
        }
        _ => {}
    }
}

fn elf_collect_impl<Elf: object::read::elf::FileHeader<Endian = Endianness>>(
    data: &[u8],
    exported: &FxHashSet<String>,
    out: &mut FxHashSet<String>,
) where
    u64: From<Elf::Word>,
{
    let Ok(header) = Elf::parse(data) else { return };
    let Ok(endian) = header.endian() else { return };
    let Ok(sections) = header.sections(endian, data) else { return };
    let Ok(symtab) = sections.symbols(endian, data, elf::SHT_SYMTAB) else { return };
    let strings = symtab.strings();

    for sym in symtab.iter() {
        let binding = sym.st_bind();
        if binding != elf::STB_GLOBAL && binding != elf::STB_WEAK {
            continue;
        }
        if sym.is_undefined(endian) {
            continue;
        }
        let Ok(name_bytes) = sym.name(endian, strings) else { continue };
        let Ok(name) = str::from_utf8(name_bytes) else { continue };
        if !exported.contains(name) {
            out.insert(name.to_string());
        }
    }
}

fn macho_collect_impl<Mach: object::read::macho::MachHeader<Endian = Endianness>>(
    data: &[u8],
    exported: &FxHashSet<String>,
    out: &mut FxHashSet<String>,
) {
    let Ok(header) = Mach::parse(data, 0) else { return };
    let Ok(endian) = header.endian() else { return };
    let Ok(mut commands) = header.load_commands(endian, data, 0) else { return };

    let symtab_cmd = loop {
        let Ok(Some(cmd)) = commands.next() else { return };
        if let Ok(Some(st)) = cmd.symtab() {
            break st;
        }
    };
    let Ok(symtab) = symtab_cmd.symbols::<Mach, _>(endian, data) else { return };
    let strings = symtab.strings();

    for nlist in symtab.iter() {
        if nlist.is_stab() {
            continue;
        }
        if nlist.is_undefined() {
            continue;
        }
        if nlist.n_type() & macho::N_EXT == 0 {
            continue;
        }
        let Ok(name_bytes) = nlist.name(endian, strings) else { continue };
        let Ok(name) = str::from_utf8(name_bytes) else { continue };
        let name = name.strip_prefix('_').unwrap_or(name);
        if !exported.contains(name) {
            out.insert(name.to_string());
        }
    }
}

// ---------------------------------------------------------------------------
// ELF: single-pass collection + apply
// ---------------------------------------------------------------------------

fn elf_edit_impl<Elf: object::read::elf::FileHeader<Endian = Endianness>>(
    data: &[u8],
    exported: &FxHashSet<String>,
    hide: bool,
    rename: Option<&(FxHashSet<String>, &str)>,
    st_other_offset: usize,
) -> Option<Vec<u8>>
where
    u64: From<Elf::Word>,
{
    let header = Elf::parse(data).ok()?;
    let endian = header.endian().ok()?;
    let sections = header.sections(endian, data).ok()?;
    let symtab = sections.symbols(endian, data, elf::SHT_SYMTAB).ok()?;
    let data_ptr = data.as_ptr() as usize;
    let strings = symtab.strings();

    let mut patches = Vec::new();
    let mut renames = Vec::new();

    for sym in symtab.iter() {
        let binding = sym.st_bind();
        if binding != elf::STB_GLOBAL && binding != elf::STB_WEAK {
            continue;
        }
        let Ok(name_bytes) = sym.name(endian, strings) else { continue };
        let Ok(name) = str::from_utf8(name_bytes) else { continue };

        let sym_addr = sym as *const Elf::Sym as usize;

        if hide && !sym.is_undefined(endian) && !exported.contains(name) {
            let offset = sym_addr - data_ptr + st_other_offset;
            let new_vis = (sym.st_other() & !0x03) | elf::STV_HIDDEN;
            patches.push(Patch { offset, value: new_vis });
        }
        if let Some((rename_set, _)) = rename {
            if rename_set.contains(name) {
                renames.push(RenameEntry {
                    name_field_offset: sym_addr - data_ptr,
                    name: name.to_string(),
                });
            }
        }
    }

    if patches.is_empty() && renames.is_empty() {
        return None;
    }

    let mut result = data.to_vec();
    for p in &patches {
        result[p.offset] = p.value;
    }

    if !renames.is_empty() {
        let suffix = rename.unwrap().1;
        if let Some(renamed) =
            elf_rebuild_strtab::<Elf>(&result, &renames, suffix, &sections, header, endian)
        {
            result = renamed;
        }
    }

    Some(result)
}

fn elf_rebuild_strtab<Elf: object::read::elf::FileHeader<Endian = Endianness>>(
    data: &[u8],
    renames: &[RenameEntry],
    suffix: &str,
    sections: &object::read::elf::SectionTable<'_, Elf>,
    header: &Elf,
    endian: Endianness,
) -> Option<Vec<u8>>
where
    u64: From<Elf::Word>,
{
    let mut strtab_si: Option<usize> = None;
    for section in sections.iter() {
        if section.sh_type(endian) == elf::SHT_SYMTAB {
            strtab_si = Some(section.sh_link(endian) as usize);
            break;
        }
    }
    let strtab_si = strtab_si?;

    let e_shoff = u64::from(header.e_shoff(endian)) as usize;
    let e_shentsize = mem::size_of::<Elf::SectionHeader>();
    let e_shnum = sections.len();

    let strtab_section = sections.section(object::SectionIndex(strtab_si)).ok()?;
    let old_strtab_offset = u64::from(strtab_section.sh_offset(endian)) as usize;
    let old_strtab_size = u64::from(strtab_section.sh_size(endian)) as usize;
    let old_strtab = data.get(old_strtab_offset..old_strtab_offset + old_strtab_size)?;

    let (new_strtab, rename_map) = build_renamed_strtab(old_strtab, renames, suffix);

    let is_64 = mem::size_of::<Elf::Word>() == 8;
    let new_strtab_file_off = data.len();
    let new_strtab_size = new_strtab.len();
    let new_e_shoff_raw = new_strtab_file_off + new_strtab_size;
    let new_e_shoff = (new_e_shoff_raw + 7) & !7;
    let padding = new_e_shoff - new_e_shoff_raw;
    let section_headers_size = e_shentsize * e_shnum;

    let result_size = new_e_shoff + section_headers_size;
    let mut result = Vec::with_capacity(result_size);
    result.extend_from_slice(data);
    result.extend_from_slice(&new_strtab);
    result.resize(result.len() + padding, 0);
    let sh_data = data.get(e_shoff..e_shoff + section_headers_size)?;
    result.extend_from_slice(sh_data);

    if is_64 {
        write_u64_at(
            &mut result,
            mem::offset_of!(elf::FileHeader64<Endianness>, e_shoff),
            new_e_shoff as u64,
            endian,
        );
    } else {
        write_u32_at(
            &mut result,
            mem::offset_of!(elf::FileHeader32<Endianness>, e_shoff),
            new_e_shoff as u32,
            endian,
        );
    }

    let new_strtab_shdr_offset = new_e_shoff + strtab_si * e_shentsize;

    if is_64 {
        let sh_offset_field = mem::offset_of!(elf::SectionHeader64<Endianness>, sh_offset);
        let sh_size_field = mem::offset_of!(elf::SectionHeader64<Endianness>, sh_size);
        write_u64_at(
            &mut result,
            new_strtab_shdr_offset + sh_offset_field,
            new_strtab_file_off as u64,
            endian,
        );
        write_u64_at(
            &mut result,
            new_strtab_shdr_offset + sh_size_field,
            new_strtab_size as u64,
            endian,
        );
    } else {
        let sh_offset_field = mem::offset_of!(elf::SectionHeader32<Endianness>, sh_offset);
        let sh_size_field = mem::offset_of!(elf::SectionHeader32<Endianness>, sh_size);
        write_u32_at(
            &mut result,
            new_strtab_shdr_offset + sh_offset_field,
            new_strtab_file_off as u32,
            endian,
        );
        write_u32_at(
            &mut result,
            new_strtab_shdr_offset + sh_size_field,
            new_strtab_size as u32,
            endian,
        );
    }

    for entry in renames {
        let new_st_name = rename_map[&entry.name];
        write_u32_at(&mut result, entry.name_field_offset, new_st_name, endian);
    }

    Some(result)
}

// ---------------------------------------------------------------------------
// Mach-O: single-pass collection + apply
// ---------------------------------------------------------------------------

fn macho_edit_impl<Mach: object::read::macho::MachHeader<Endian = Endianness>>(
    data: &[u8],
    exported: &FxHashSet<String>,
    hide: bool,
    rename: Option<&(FxHashSet<String>, &str)>,
    n_type_offset: usize,
) -> Option<Vec<u8>> {
    let header = Mach::parse(data, 0).ok()?;
    let endian = header.endian().ok()?;
    let mut commands = header.load_commands(endian, data, 0).ok()?;

    let (symtab_cmd, symtab_cmd_offset) = loop {
        let cmd = commands.next().ok()??;
        if let Some(st) = cmd.symtab().ok().flatten() {
            break (st, cmd.raw_data().as_ptr() as usize - data.as_ptr() as usize);
        }
    };

    let symtab: object::read::macho::SymbolTable<'_, Mach, &_> =
        symtab_cmd.symbols(endian, data).ok()?;
    let data_ptr = data.as_ptr() as usize;
    let strings = symtab.strings();

    let mut patches = Vec::new();
    let mut renames = Vec::new();

    for nlist in symtab.iter() {
        if nlist.is_stab() {
            continue;
        }
        if nlist.n_type() & macho::N_EXT == 0 {
            continue;
        }
        let Ok(name_bytes) = nlist.name(endian, strings) else { continue };
        let Ok(raw_name) = str::from_utf8(name_bytes) else { continue };
        let name = raw_name.strip_prefix('_').unwrap_or(raw_name);

        let nlist_addr = nlist as *const Mach::Nlist as usize;

        if hide && !nlist.is_undefined() && !exported.contains(name) {
            let offset = nlist_addr - data_ptr + n_type_offset;
            patches.push(Patch { offset, value: nlist.n_type() | macho::N_PEXT });
        }
        if let Some((rename_set, _)) = rename {
            if rename_set.contains(name) {
                renames.push(RenameEntry {
                    name_field_offset: nlist_addr - data_ptr,
                    name: raw_name.to_string(),
                });
            }
        }
    }

    if patches.is_empty() && renames.is_empty() {
        return None;
    }

    let mut result = data.to_vec();
    for p in &patches {
        result[p.offset] = p.value;
    }

    if !renames.is_empty() {
        let suffix = rename.unwrap().1;
        if let Some(renamed) =
            macho_rebuild_strtab(&result, &renames, suffix, &symtab_cmd, symtab_cmd_offset, endian)
        {
            result = renamed;
        }
    }

    Some(result)
}

fn macho_rebuild_strtab(
    data: &[u8],
    renames: &[RenameEntry],
    suffix: &str,
    symtab_cmd: &macho::SymtabCommand<Endianness>,
    symtab_cmd_offset: usize,
    endian: Endianness,
) -> Option<Vec<u8>> {
    let old_stroff = symtab_cmd.stroff.get(endian) as usize;
    let old_strsize = symtab_cmd.strsize.get(endian) as usize;
    let old_strtab = data.get(old_stroff..old_stroff + old_strsize)?;

    let (new_strtab, rename_map) = build_renamed_strtab(old_strtab, renames, suffix);

    let new_strtab_file_off = data.len();
    let new_strtab_size = new_strtab.len();

    let mut result = Vec::with_capacity(data.len() + new_strtab_size);
    result.extend_from_slice(data);
    result.extend_from_slice(&new_strtab);

    let stroff_off = mem::offset_of!(macho::SymtabCommand<Endianness>, stroff);
    let strsize_off = mem::offset_of!(macho::SymtabCommand<Endianness>, strsize);
    write_u32_at(&mut result, symtab_cmd_offset + stroff_off, new_strtab_file_off as u32, endian);
    write_u32_at(&mut result, symtab_cmd_offset + strsize_off, new_strtab_size as u32, endian);

    for entry in renames {
        let new_strx = rename_map[&entry.name];
        write_u32_at(&mut result, entry.name_field_offset, new_strx, endian);
    }

    Some(result)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn build_renamed_strtab(
    old_strtab: &[u8],
    renames: &[RenameEntry],
    suffix: &str,
) -> (Vec<u8>, FxHashMap<String, u32>) {
    let mut new_strtab = old_strtab.to_vec();
    let mut rename_map: FxHashMap<String, u32> = FxHashMap::default();

    let mut sorted_names: Vec<&str> = renames.iter().map(|r| r.name.as_str()).collect();
    sorted_names.sort();
    sorted_names.dedup();

    for name in &sorted_names {
        let new_offset = new_strtab.len() as u32;
        new_strtab.extend_from_slice(name.as_bytes());
        new_strtab.extend_from_slice(suffix.as_bytes());
        new_strtab.push(0);
        rename_map.insert(name.to_string(), new_offset);
    }

    (new_strtab, rename_map)
}

fn write_u32_at(buf: &mut [u8], offset: usize, value: u32, endian: Endianness) {
    let bytes = match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    };
    buf[offset..offset + 4].copy_from_slice(&bytes);
}

fn write_u64_at(buf: &mut [u8], offset: usize, value: u64, endian: Endianness) {
    let bytes = match endian {
        Endianness::Little => value.to_le_bytes(),
        Endianness::Big => value.to_be_bytes(),
    };
    buf[offset..offset + 8].copy_from_slice(&bytes);
}
