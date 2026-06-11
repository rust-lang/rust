// We use the `object` crate for the read-only pass over ELF/Mach-O object files
// because its `Sym`/`Nlist` traits provide clean access to symbol properties without
// manual byte parsing. However, `object` does not expose mutable views into the data,
// so we cannot use it to modify symbol fields in place. Instead, the read-only pass
// collects byte-level patches (offset + new value), and the write pass
// (`apply_patches`) applies them to a copy of the byte buffer without any ELF/Mach-O
// parsing — similar to how linker relocations work.

use std::mem;

use object::read::elf::Sym as _;
use object::read::macho::Nlist;
use object::{Endianness, elf, macho};
use rustc_data_structures::fx::FxHashSet;

/// A byte-level patch collected in the read-only pass and applied in the write pass.
struct Patch {
    offset: usize,
    value: u8,
}

/// Apply a list of byte patches to `data`, returning the (possibly modified) bytes.
fn apply_patches(data: &[u8], patches: &[Patch]) -> Vec<u8> {
    let mut buf = data.to_vec();
    for p in patches {
        buf[p.offset] = p.value;
    }
    buf
}

// ---------------------------------------------------------------------------
// ELF hide – read-only pass uses `object` crate, write pass uses `Patch` list
// ---------------------------------------------------------------------------

fn elf_hide_patches_impl<'data, Elf: object::read::elf::FileHeader<Endian = Endianness>>(
    data: &'data [u8],
    st_other_offset: usize,
    exported: &FxHashSet<String>,
) -> Option<Vec<Patch>>
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
            let sym_addr = sym as *const Elf::Sym as usize;
            let offset = sym_addr - data_ptr + st_other_offset;
            let new_vis = (sym.st_other() & !0x03) | elf::STV_HIDDEN;
            patches.push(Patch { offset, value: new_vis });
        }
    }

    Some(patches)
}

// ---------------------------------------------------------------------------
// Mach-O hide – same architecture: read-only pass via `object`, write via patches
// ---------------------------------------------------------------------------

fn macho_hide_patches_impl<'data, Mach: object::read::macho::MachHeader<Endian = Endianness>>(
    data: &'data [u8],
    n_type_offset: usize,
    exported: &FxHashSet<String>,
) -> Option<Vec<Patch>> {
    let header = Mach::parse(data, 0).ok()?;
    let endian = header.endian().ok()?;
    let mut commands = header.load_commands(endian, data, 0).ok()?;

    let symtab_cmd = loop {
        let cmd = commands.next().ok()??;
        if let Some(st) = cmd.symtab().ok().flatten() {
            break st;
        }
    };
    let symtab: object::read::macho::SymbolTable<'_, Mach, &_> =
        symtab_cmd.symbols(endian, data).ok()?;

    let data_ptr = data.as_ptr() as usize;
    let strings = symtab.strings();
    let mut patches = Vec::new();

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
            let nlist_addr = nlist as *const Mach::Nlist as usize;
            let offset = nlist_addr - data_ptr + n_type_offset;
            patches.push(Patch { offset, value: nlist.n_type() | macho::N_PEXT });
        }
    }

    Some(patches)
}

// ---------------------------------------------------------------------------
// Unified dispatch: top-level detection via `object::File::parse`
// ---------------------------------------------------------------------------

fn hide_patches(data: &[u8], exported: &FxHashSet<String>) -> Option<Vec<Patch>> {
    let file = object::File::parse(data).ok()?;
    match file {
        object::File::Elf64(_) => elf_hide_patches_impl::<elf::FileHeader64<Endianness>>(
            data,
            mem::offset_of!(elf::Sym64<Endianness>, st_other),
            exported,
        ),
        object::File::Elf32(_) => elf_hide_patches_impl::<elf::FileHeader32<Endianness>>(
            data,
            mem::offset_of!(elf::Sym32<Endianness>, st_other),
            exported,
        ),
        object::File::MachO64(_) => macho_hide_patches_impl::<macho::MachHeader64<Endianness>>(
            data,
            mem::offset_of!(macho::Nlist64<Endianness>, n_type),
            exported,
        ),
        object::File::MachO32(_) => macho_hide_patches_impl::<macho::MachHeader32<Endianness>>(
            data,
            mem::offset_of!(macho::Nlist32<Endianness>, n_type),
            exported,
        ),
        _ => None,
    }
}

pub(super) fn apply_hide(data: &[u8], exported: &FxHashSet<String>) -> Vec<u8> {
    let patches = hide_patches(data, exported).unwrap_or_default();
    apply_patches(data, &patches)
}
