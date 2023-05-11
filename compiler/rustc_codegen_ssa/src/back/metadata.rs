//! Reading of the rustc metadata for rlibs and dylibs

use std::fs::File;
use std::io::Write;
use std::path::Path;

use object::write::{self, StandardSegment, Symbol, SymbolSection};
use object::{
    elf, pe, Architecture, BinaryFormat, Endianness, FileFlags, Object, ObjectSection,
    SectionFlags, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};

use snap::write::FrameEncoder;

use object::elf::NT_GNU_PROPERTY_TYPE_0;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owned_slice::try_slice_owned;
use rustc_data_structures::sync::MetadataRef;
use rustc_metadata::fs::METADATA_FILENAME;
use rustc_metadata::EncodedMetadata;
use rustc_session::cstore::MetadataLoader;
use rustc_session::Session;
use rustc_target::abi::Endian;
use rustc_target::spec::{RelocModel, Target};

/// The default metadata loader. This is used by cg_llvm and cg_clif.
///
/// # Metadata location
///
/// <dl>
/// <dt>rlib</dt>
/// <dd>The metadata can be found in the `lib.rmeta` file inside of the ar archive.</dd>
/// <dt>dylib</dt>
/// <dd>The metadata can be found in the `.rustc` section of the shared library.</dd>
/// </dl>
#[derive(Debug)]
pub struct DefaultMetadataLoader;

fn load_metadata_with(
    path: &Path,
    f: impl for<'a> FnOnce(&'a [u8]) -> Result<&'a [u8], String>,
) -> Result<MetadataRef, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file '{}': {}", path.display(), e))?;

    unsafe { Mmap::map(file) }
        .map_err(|e| format!("failed to mmap file '{}': {}", path.display(), e))
        .and_then(|mmap| try_slice_owned(mmap, |mmap| f(mmap)))
}

impl MetadataLoader for DefaultMetadataLoader {
    fn get_rlib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        load_metadata_with(path, |data| {
            let archive = object::read::archive::ArchiveFile::parse(&*data)
                .map_err(|e| format!("failed to parse rlib '{}': {}", path.display(), e))?;

            for entry_result in archive.members() {
                let entry = entry_result
                    .map_err(|e| format!("failed to parse rlib '{}': {}", path.display(), e))?;
                if entry.name() == METADATA_FILENAME.as_bytes() {
                    let data = entry
                        .data(data)
                        .map_err(|e| format!("failed to parse rlib '{}': {}", path.display(), e))?;
                    return search_for_section(path, data, ".rmeta");
                }
            }

            Err(format!("metadata not found in rlib '{}'", path.display()))
        })
    }

    fn get_dylib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        load_metadata_with(path, |data| search_for_section(path, data, ".rustc"))
    }
}

pub(super) fn search_for_section<'a>(
    path: &Path,
    bytes: &'a [u8],
    section: &str,
) -> Result<&'a [u8], String> {
    let Ok(file) = object::File::parse(bytes) else {
        // The parse above could fail for odd reasons like corruption, but for
        // now we just interpret it as this target doesn't support metadata
        // emission in object files so the entire byte slice itself is probably
        // a metadata file. Ideally though if necessary we could at least check
        // the prefix of bytes to see if it's an actual metadata object and if
        // not forward the error along here.
        return Ok(bytes);
    };
    file.section_by_name(section)
        .ok_or_else(|| format!("no `{}` section in '{}'", section, path.display()))?
        .data()
        .map_err(|e| format!("failed to read {} section in '{}': {}", section, path.display(), e))
}

fn add_gnu_property_note(
    file: &mut write::Object<'static>,
    architecture: Architecture,
    binary_format: BinaryFormat,
    endianness: Endianness,
) {
    // check bti protection
    if binary_format != BinaryFormat::Elf
        || !matches!(architecture, Architecture::X86_64 | Architecture::Aarch64)
    {
        return;
    }

    let section = file.add_section(
        file.segment_name(StandardSegment::Data).to_vec(),
        b".note.gnu.property".to_vec(),
        SectionKind::Note,
    );
    let mut data: Vec<u8> = Vec::new();
    let n_namsz: u32 = 4; // Size of the n_name field
    let n_descsz: u32 = 16; // Size of the n_desc field
    let n_type: u32 = NT_GNU_PROPERTY_TYPE_0; // Type of note descriptor
    let header_values = [n_namsz, n_descsz, n_type];
    header_values.iter().for_each(|v| {
        data.extend_from_slice(&match endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        })
    });
    data.extend_from_slice(b"GNU\0"); // Owner of the program property note
    let pr_type: u32 = match architecture {
        Architecture::X86_64 => 0xc0000002,
        Architecture::Aarch64 => 0xc0000000,
        _ => unreachable!(),
    };
    let pr_datasz: u32 = 4; //size of the pr_data field
    let pr_data: u32 = 3; //program property descriptor
    let pr_padding: u32 = 0;
    let property_values = [pr_type, pr_datasz, pr_data, pr_padding];
    property_values.iter().for_each(|v| {
        data.extend_from_slice(&match endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        })
    });
    file.append_section_data(section, &data, 8);
}

pub(crate) fn create_object_file(sess: &Session) -> Option<write::Object<'static>> {
    let endianness = match sess.target.options.endian {
        Endian::Little => Endianness::Little,
        Endian::Big => Endianness::Big,
    };
    let architecture = match &sess.target.arch[..] {
        "arm" => Architecture::Arm,
        "aarch64" => {
            if sess.target.pointer_width == 32 {
                Architecture::Aarch64_Ilp32
            } else {
                Architecture::Aarch64
            }
        }
        "x86" => Architecture::I386,
        "s390x" => Architecture::S390x,
        "mips" => Architecture::Mips,
        "mips64" => Architecture::Mips64,
        "x86_64" => {
            if sess.target.pointer_width == 32 {
                Architecture::X86_64_X32
            } else {
                Architecture::X86_64
            }
        }
        "powerpc" => Architecture::PowerPc,
        "powerpc64" => Architecture::PowerPc64,
        "riscv32" => Architecture::Riscv32,
        "riscv64" => Architecture::Riscv64,
        "sparc64" => Architecture::Sparc64,
        "avr" => Architecture::Avr,
        "msp430" => Architecture::Msp430,
        "hexagon" => Architecture::Hexagon,
        "bpf" => Architecture::Bpf,
        "loongarch64" => Architecture::LoongArch64,
        // Unsupported architecture.
        _ => return None,
    };
    let binary_format = if sess.target.is_like_osx {
        BinaryFormat::MachO
    } else if sess.target.is_like_windows {
        BinaryFormat::Coff
    } else {
        BinaryFormat::Elf
    };

    let mut file = write::Object::new(binary_format, architecture, endianness);
    let e_flags = match architecture {
        Architecture::Mips => {
            let arch = match sess.target.options.cpu.as_ref() {
                "mips1" => elf::EF_MIPS_ARCH_1,
                "mips2" => elf::EF_MIPS_ARCH_2,
                "mips3" => elf::EF_MIPS_ARCH_3,
                "mips4" => elf::EF_MIPS_ARCH_4,
                "mips5" => elf::EF_MIPS_ARCH_5,
                s if s.contains("r6") => elf::EF_MIPS_ARCH_32R6,
                _ => elf::EF_MIPS_ARCH_32R2,
            };
            // The only ABI LLVM supports for 32-bit MIPS CPUs is o32.
            let mut e_flags = elf::EF_MIPS_CPIC | elf::EF_MIPS_ABI_O32 | arch;
            if sess.target.options.relocation_model != RelocModel::Static {
                e_flags |= elf::EF_MIPS_PIC;
            }
            if sess.target.options.cpu.contains("r6") {
                e_flags |= elf::EF_MIPS_NAN2008;
            }
            e_flags
        }
        Architecture::Mips64 => {
            // copied from `mips64el-linux-gnuabi64-gcc foo.c -c`
            let e_flags = elf::EF_MIPS_CPIC
                | elf::EF_MIPS_PIC
                | if sess.target.options.cpu.contains("r6") {
                    elf::EF_MIPS_ARCH_64R6 | elf::EF_MIPS_NAN2008
                } else {
                    elf::EF_MIPS_ARCH_64R2
                };
            e_flags
        }
        Architecture::Riscv32 | Architecture::Riscv64 => {
            // Source: https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/079772828bd10933d34121117a222b4cc0ee2200/riscv-elf.adoc
            let mut e_flags: u32 = 0x0;
            let features = &sess.target.options.features;
            // Check if compressed is enabled
            if features.contains("+c") {
                e_flags |= elf::EF_RISCV_RVC;
            }

            // Select the appropriate floating-point ABI
            if features.contains("+d") {
                e_flags |= elf::EF_RISCV_FLOAT_ABI_DOUBLE;
            } else if features.contains("+f") {
                e_flags |= elf::EF_RISCV_FLOAT_ABI_SINGLE;
            } else {
                e_flags |= elf::EF_RISCV_FLOAT_ABI_SOFT;
            }
            e_flags
        }
        Architecture::LoongArch64 => {
            // Source: https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html#_e_flags_identifies_abi_type_and_version
            elf::EF_LARCH_OBJABI_V1 | elf::EF_LARCH_ABI_DOUBLE_FLOAT
        }
        _ => 0,
    };
    // adapted from LLVM's `MCELFObjectTargetWriter::getOSABI`
    let os_abi = match sess.target.options.os.as_ref() {
        "hermit" => elf::ELFOSABI_STANDALONE,
        "freebsd" => elf::ELFOSABI_FREEBSD,
        "solaris" => elf::ELFOSABI_SOLARIS,
        _ => elf::ELFOSABI_NONE,
    };
    let abi_version = 0;
    add_gnu_property_note(&mut file, architecture, binary_format, endianness);
    file.flags = FileFlags::Elf { os_abi, abi_version, e_flags };
    Some(file)
}

pub enum MetadataPosition {
    First,
    Last,
}

/// For rlibs we "pack" rustc metadata into a dummy object file.
///
/// Historically it was needed because rustc linked rlibs as whole-archive in some cases.
/// In that case linkers try to include all files located in an archive, so if metadata is stored
/// in an archive then it needs to be of a form that the linker is able to process.
/// Now it's not clear whether metadata still needs to be wrapped into an object file or not.
///
/// Note, though, that we don't actually want this metadata to show up in any
/// final output of the compiler. Instead this is purely for rustc's own
/// metadata tracking purposes.
///
/// With the above in mind, each "flavor" of object format gets special
/// handling here depending on the target:
///
/// * MachO - macos-like targets will insert the metadata into a section that
///   is sort of fake dwarf debug info. Inspecting the source of the macos
///   linker this causes these sections to be skipped automatically because
///   it's not in an allowlist of otherwise well known dwarf section names to
///   go into the final artifact.
///
/// * WebAssembly - we actually don't have any container format for this
///   target. WebAssembly doesn't support the `dylib` crate type anyway so
///   there's no need for us to support this at this time. Consequently the
///   metadata bytes are simply stored as-is into an rlib.
///
/// * COFF - Windows-like targets create an object with a section that has
///   the `IMAGE_SCN_LNK_REMOVE` flag set which ensures that if the linker
///   ever sees the section it doesn't process it and it's removed.
///
/// * ELF - All other targets are similar to Windows in that there's a
///   `SHF_EXCLUDE` flag we can set on sections in an object file to get
///   automatically removed from the final output.
pub fn create_wrapper_file(
    sess: &Session,
    section_name: Vec<u8>,
    data: &[u8],
) -> (Vec<u8>, MetadataPosition) {
    let Some(mut file) = create_object_file(sess) else {
        // This is used to handle all "other" targets. This includes targets
        // in two categories:
        //
        // * Some targets don't have support in the `object` crate just yet
        //   to write an object file. These targets are likely to get filled
        //   out over time.
        //
        // * Targets like WebAssembly don't support dylibs, so the purpose
        //   of putting metadata in object files, to support linking rlibs
        //   into dylibs, is moot.
        //
        // In both of these cases it means that linking into dylibs will
        // not be supported by rustc. This doesn't matter for targets like
        // WebAssembly and for targets not supported by the `object` crate
        // yet it means that work will need to be done in the `object` crate
        // to add a case above.
        return (data.to_vec(), MetadataPosition::Last);
    };
    let section = file.add_section(
        file.segment_name(StandardSegment::Debug).to_vec(),
        section_name,
        SectionKind::Debug,
    );
    match file.format() {
        BinaryFormat::Coff => {
            file.section_mut(section).flags =
                SectionFlags::Coff { characteristics: pe::IMAGE_SCN_LNK_REMOVE };
        }
        BinaryFormat::Elf => {
            file.section_mut(section).flags =
                SectionFlags::Elf { sh_flags: elf::SHF_EXCLUDE as u64 };
        }
        _ => {}
    };
    file.append_section_data(section, data, 1);
    (file.write().unwrap(), MetadataPosition::First)
}

// Historical note:
//
// When using link.exe it was seen that the section name `.note.rustc`
// was getting shortened to `.note.ru`, and according to the PE and COFF
// specification:
//
// > Executable images do not use a string table and do not support
// > section names longer than 8 characters
//
// https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
//
// As a result, we choose a slightly shorter name! As to why
// `.note.rustc` works on MinGW, see
// https://github.com/llvm/llvm-project/blob/llvmorg-12.0.0/lld/COFF/Writer.cpp#L1190-L1197
pub fn create_compressed_metadata_file(
    sess: &Session,
    metadata: &EncodedMetadata,
    symbol_name: &str,
) -> Vec<u8> {
    let mut compressed = rustc_metadata::METADATA_HEADER.to_vec();
    // Our length will be backfilled once we're done writing
    compressed.write_all(&[0; 4]).unwrap();
    FrameEncoder::new(&mut compressed).write_all(metadata.raw_data()).unwrap();
    let meta_len = rustc_metadata::METADATA_HEADER.len();
    let data_len = (compressed.len() - meta_len - 4) as u32;
    compressed[meta_len..meta_len + 4].copy_from_slice(&data_len.to_be_bytes());

    let Some(mut file) = create_object_file(sess) else {
        return compressed.to_vec();
    };
    let section = file.add_section(
        file.segment_name(StandardSegment::Data).to_vec(),
        b".rustc".to_vec(),
        SectionKind::ReadOnlyData,
    );
    match file.format() {
        BinaryFormat::Elf => {
            // Explicitly set no flags to avoid SHF_ALLOC default for data section.
            file.section_mut(section).flags = SectionFlags::Elf { sh_flags: 0 };
        }
        _ => {}
    };
    let offset = file.append_section_data(section, &compressed, 1);

    // For MachO and probably PE this is necessary to prevent the linker from throwing away the
    // .rustc section. For ELF this isn't necessary, but it also doesn't harm.
    file.add_symbol(Symbol {
        name: symbol_name.as_bytes().to_vec(),
        value: offset,
        size: compressed.len() as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });

    file.write().unwrap()
}
