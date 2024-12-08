//! Reading of the rustc metadata for rlibs and dylibs

use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use itertools::Itertools;
use object::write::{self, StandardSegment, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, FileFlags, Object, ObjectSection, ObjectSymbol,
    SectionFlags, SectionKind, SubArchitecture, SymbolFlags, SymbolKind, SymbolScope, elf, pe,
    xcoff,
};
use rustc_abi::Endian;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owned_slice::{OwnedSlice, try_slice_owned};
use rustc_metadata::EncodedMetadata;
use rustc_metadata::creader::MetadataLoader;
use rustc_metadata::fs::METADATA_FILENAME;
use rustc_middle::bug;
use rustc_session::Session;
use rustc_span::sym;
use rustc_target::spec::{RelocModel, Target, ef_avr_arch};
use tracing::debug;

use super::apple;

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
pub(crate) struct DefaultMetadataLoader;

static AIX_METADATA_SYMBOL_NAME: &'static str = "__aix_rust_metadata";

fn load_metadata_with(
    path: &Path,
    f: impl for<'a> FnOnce(&'a [u8]) -> Result<&'a [u8], String>,
) -> Result<OwnedSlice, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file '{}': {}", path.display(), e))?;

    unsafe { Mmap::map(file) }
        .map_err(|e| format!("failed to mmap file '{}': {}", path.display(), e))
        .and_then(|mmap| try_slice_owned(mmap, |mmap| f(mmap)))
}

impl MetadataLoader for DefaultMetadataLoader {
    fn get_rlib_metadata(&self, target: &Target, path: &Path) -> Result<OwnedSlice, String> {
        debug!("getting rlib metadata for {}", path.display());
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
                    if target.is_like_aix {
                        return get_metadata_xcoff(path, data);
                    } else {
                        return search_for_section(path, data, ".rmeta");
                    }
                }
            }

            Err(format!("metadata not found in rlib '{}'", path.display()))
        })
    }

    fn get_dylib_metadata(&self, target: &Target, path: &Path) -> Result<OwnedSlice, String> {
        debug!("getting dylib metadata for {}", path.display());
        if target.is_like_aix {
            load_metadata_with(path, |data| {
                let archive = object::read::archive::ArchiveFile::parse(&*data).map_err(|e| {
                    format!("failed to parse aix dylib '{}': {}", path.display(), e)
                })?;

                match archive.members().exactly_one() {
                    Ok(lib) => {
                        let lib = lib.map_err(|e| {
                            format!("failed to parse aix dylib '{}': {}", path.display(), e)
                        })?;
                        let data = lib.data(data).map_err(|e| {
                            format!("failed to parse aix dylib '{}': {}", path.display(), e)
                        })?;
                        get_metadata_xcoff(path, data)
                    }
                    Err(e) => Err(format!("failed to parse aix dylib '{}': {}", path.display(), e)),
                }
            })
        } else {
            load_metadata_with(path, |data| search_for_section(path, data, ".rustc"))
        }
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
    let n_type: u32 = object::elf::NT_GNU_PROPERTY_TYPE_0; // Type of note descriptor
    let header_values = [n_namsz, n_descsz, n_type];
    header_values.iter().for_each(|v| {
        data.extend_from_slice(&match endianness {
            Endianness::Little => v.to_le_bytes(),
            Endianness::Big => v.to_be_bytes(),
        })
    });
    data.extend_from_slice(b"GNU\0"); // Owner of the program property note
    let pr_type: u32 = match architecture {
        Architecture::X86_64 => object::elf::GNU_PROPERTY_X86_FEATURE_1_AND,
        Architecture::Aarch64 => object::elf::GNU_PROPERTY_AARCH64_FEATURE_1_AND,
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

pub(super) fn get_metadata_xcoff<'a>(path: &Path, data: &'a [u8]) -> Result<&'a [u8], String> {
    let Ok(file) = object::File::parse(data) else {
        return Ok(data);
    };
    let info_data = search_for_section(path, data, ".info")?;
    if let Some(metadata_symbol) =
        file.symbols().find(|sym| sym.name() == Ok(AIX_METADATA_SYMBOL_NAME))
    {
        let offset = metadata_symbol.address() as usize;
        // The offset specifies the location of rustc metadata in the .info section of XCOFF.
        // Each string stored in .info section of XCOFF is preceded by a 4-byte length field.
        if offset < 4 {
            return Err(format!("Invalid metadata symbol offset: {offset}"));
        }
        // XCOFF format uses big-endian byte order.
        let len = u32::from_be_bytes(info_data[(offset - 4)..offset].try_into().unwrap()) as usize;
        if offset + len > (info_data.len() as usize) {
            return Err(format!(
                "Metadata at offset {offset} with size {len} is beyond .info section"
            ));
        }
        Ok(&info_data[offset..(offset + len)])
    } else {
        Err(format!("Unable to find symbol {AIX_METADATA_SYMBOL_NAME}"))
    }
}

pub(crate) fn create_object_file(sess: &Session) -> Option<write::Object<'static>> {
    let endianness = match sess.target.options.endian {
        Endian::Little => Endianness::Little,
        Endian::Big => Endianness::Big,
    };
    let (architecture, sub_architecture) = match &sess.target.arch[..] {
        "arm" => (Architecture::Arm, None),
        "aarch64" => (
            if sess.target.pointer_width == 32 {
                Architecture::Aarch64_Ilp32
            } else {
                Architecture::Aarch64
            },
            None,
        ),
        "x86" => (Architecture::I386, None),
        "s390x" => (Architecture::S390x, None),
        "mips" | "mips32r6" => (Architecture::Mips, None),
        "mips64" | "mips64r6" => (Architecture::Mips64, None),
        "x86_64" => (
            if sess.target.pointer_width == 32 {
                Architecture::X86_64_X32
            } else {
                Architecture::X86_64
            },
            None,
        ),
        "powerpc" => (Architecture::PowerPc, None),
        "powerpc64" => (Architecture::PowerPc64, None),
        "riscv32" => (Architecture::Riscv32, None),
        "riscv64" => (Architecture::Riscv64, None),
        "sparc" => {
            if sess.unstable_target_features.contains(&sym::v8plus) {
                // Target uses V8+, aka EM_SPARC32PLUS, aka 64-bit V9 but in 32-bit mode
                (Architecture::Sparc32Plus, None)
            } else {
                // Target uses V7 or V8, aka EM_SPARC
                (Architecture::Sparc, None)
            }
        }
        "sparc64" => (Architecture::Sparc64, None),
        "avr" => (Architecture::Avr, None),
        "msp430" => (Architecture::Msp430, None),
        "hexagon" => (Architecture::Hexagon, None),
        "bpf" => (Architecture::Bpf, None),
        "loongarch64" => (Architecture::LoongArch64, None),
        "csky" => (Architecture::Csky, None),
        "arm64ec" => (Architecture::Aarch64, Some(SubArchitecture::Arm64EC)),
        // Unsupported architecture.
        _ => return None,
    };
    let binary_format = if sess.target.is_like_osx {
        BinaryFormat::MachO
    } else if sess.target.is_like_windows {
        BinaryFormat::Coff
    } else if sess.target.is_like_aix {
        BinaryFormat::Xcoff
    } else {
        BinaryFormat::Elf
    };

    let mut file = write::Object::new(binary_format, architecture, endianness);
    file.set_sub_architecture(sub_architecture);
    if sess.target.is_like_osx {
        if macho_is_arm64e(&sess.target) {
            file.set_macho_cpu_subtype(object::macho::CPU_SUBTYPE_ARM64E);
        }

        file.set_macho_build_version(macho_object_build_version_for_target(sess))
    }
    if binary_format == BinaryFormat::Coff {
        // Disable the default mangler to avoid mangling the special "@feat.00" symbol name.
        let original_mangling = file.mangling();
        file.set_mangling(object::write::Mangling::None);

        let mut feature = 0;

        if file.architecture() == object::Architecture::I386 {
            // When linking with /SAFESEH on x86, lld requires that all linker inputs be marked as
            // safe exception handling compatible. Metadata files masquerade as regular COFF
            // objects and are treated as linker inputs, despite containing no actual code. Thus,
            // they still need to be marked as safe exception handling compatible. See #96498.
            // Reference: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
            feature |= 1;
        }

        file.add_symbol(object::write::Symbol {
            name: "@feat.00".into(),
            value: feature,
            size: 0,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: object::write::SymbolSection::Absolute,
            flags: object::SymbolFlags::None,
        });

        file.set_mangling(original_mangling);
    }
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

            let mut e_flags = elf::EF_MIPS_CPIC | arch;

            // If the ABI is explicitly given, use it or default to O32.
            match sess.target.options.llvm_abiname.to_lowercase().as_str() {
                "n32" => e_flags |= elf::EF_MIPS_ABI2,
                "o32" => e_flags |= elf::EF_MIPS_ABI_O32,
                _ => e_flags |= elf::EF_MIPS_ABI_O32,
            };

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

            // Check if compressed is enabled
            // `unstable_target_features` is used here because "c" is gated behind riscv_target_feature.
            if sess.unstable_target_features.contains(&sym::c) {
                e_flags |= elf::EF_RISCV_RVC;
            }

            // Set the appropriate flag based on ABI
            // This needs to match LLVM `RISCVELFStreamer.cpp`
            match &*sess.target.llvm_abiname {
                "ilp32" | "lp64" => (),
                "ilp32f" | "lp64f" => e_flags |= elf::EF_RISCV_FLOAT_ABI_SINGLE,
                "ilp32d" | "lp64d" => e_flags |= elf::EF_RISCV_FLOAT_ABI_DOUBLE,
                // Note that the `lp64e` is still unstable as it's not (yet) part of the ELF psABI.
                "ilp32e" | "lp64e" => e_flags |= elf::EF_RISCV_RVE,
                _ => bug!("unknown RISC-V ABI name"),
            }

            e_flags
        }
        Architecture::LoongArch64 => {
            // Source: https://github.com/loongson/la-abi-specs/blob/release/laelf.adoc#e_flags-identifies-abi-type-and-version
            let mut e_flags: u32 = elf::EF_LARCH_OBJABI_V1;

            // Set the appropriate flag based on ABI
            // This needs to match LLVM `LoongArchELFStreamer.cpp`
            match &*sess.target.llvm_abiname {
                "ilp32s" | "lp64s" => e_flags |= elf::EF_LARCH_ABI_SOFT_FLOAT,
                "ilp32f" | "lp64f" => e_flags |= elf::EF_LARCH_ABI_SINGLE_FLOAT,
                "ilp32d" | "lp64d" => e_flags |= elf::EF_LARCH_ABI_DOUBLE_FLOAT,
                _ => bug!("unknown LoongArch ABI name"),
            }

            e_flags
        }
        Architecture::Avr => {
            // Resolve the ISA revision and set
            // the appropriate EF_AVR_ARCH flag.
            ef_avr_arch(&sess.target.options.cpu)
        }
        Architecture::Csky => {
            let e_flags = match sess.target.options.abi.as_ref() {
                "abiv2" => elf::EF_CSKY_ABIV2,
                _ => elf::EF_CSKY_ABIV1,
            };
            e_flags
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

/// Mach-O files contain information about:
/// - The platform/OS they were built for (macOS/watchOS/Mac Catalyst/iOS simulator etc).
/// - The minimum OS version / deployment target.
/// - The version of the SDK they were targetting.
///
/// In the past, this was accomplished using the LC_VERSION_MIN_MACOSX, LC_VERSION_MIN_IPHONEOS,
/// LC_VERSION_MIN_TVOS or LC_VERSION_MIN_WATCHOS load commands, which each contain information
/// about the deployment target and SDK version, and implicitly, by their presence, which OS they
/// target. Simulator targets were determined if the architecture was x86_64, but there was e.g. a
/// LC_VERSION_MIN_IPHONEOS present.
///
/// This is of course brittle and limited, so modern tooling emit the LC_BUILD_VERSION load
/// command (which contains all three pieces of information in one) when the deployment target is
/// high enough, or the target is something that wouldn't be encodable with the old load commands
/// (such as Mac Catalyst, or Aarch64 iOS simulator).
///
/// Since Xcode 15, Apple's LD apparently requires object files to use this load command, so this
/// returns the `MachOBuildVersion` for the target to do so.
fn macho_object_build_version_for_target(sess: &Session) -> object::write::MachOBuildVersion {
    /// The `object` crate demands "X.Y.Z encoded in nibbles as xxxx.yy.zz"
    /// e.g. minOS 14.0 = 0x000E0000, or SDK 16.2 = 0x00100200
    fn pack_version((major, minor, patch): (u16, u8, u8)) -> u32 {
        let (major, minor, patch) = (major as u32, minor as u32, patch as u32);
        (major << 16) | (minor << 8) | patch
    }

    let platform = apple::macho_platform(&sess.target);
    let min_os = apple::deployment_target(sess);

    let mut build_version = object::write::MachOBuildVersion::default();
    build_version.platform = platform;
    build_version.minos = pack_version(min_os);
    // The version here does not _really_ matter, since it is only used at runtime, and we specify
    // it when linking the final binary, so we will omit the version. This is also what LLVM does,
    // and the tooling also allows this (and shows the SDK version as `n/a`). Finally, it is the
    // semantically correct choice, as the SDK has not influenced the binary generated by rustc at
    // this point in time.
    build_version.sdk = 0;

    build_version
}

/// Is Apple's CPU subtype `arm64e`s
fn macho_is_arm64e(target: &Target) -> bool {
    target.llvm_target.starts_with("arm64e")
}

pub(crate) enum MetadataPosition {
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
/// * WebAssembly - this uses wasm files themselves as the object file format
///   so an empty file with no linking metadata but a single custom section is
///   created holding our metadata.
///
/// * COFF - Windows-like targets create an object with a section that has
///   the `IMAGE_SCN_LNK_REMOVE` flag set which ensures that if the linker
///   ever sees the section it doesn't process it and it's removed.
///
/// * ELF - All other targets are similar to Windows in that there's a
///   `SHF_EXCLUDE` flag we can set on sections in an object file to get
///   automatically removed from the final output.
pub(crate) fn create_wrapper_file(
    sess: &Session,
    section_name: String,
    data: &[u8],
) -> (Vec<u8>, MetadataPosition) {
    let Some(mut file) = create_object_file(sess) else {
        if sess.target.is_like_wasm {
            return (
                create_metadata_file_for_wasm(sess, data, &section_name),
                MetadataPosition::First,
            );
        }

        // Targets using this branch don't have support implemented here yet or
        // they're not yet implemented in the `object` crate and will likely
        // fill out this module over time.
        return (data.to_vec(), MetadataPosition::Last);
    };
    let section = if file.format() == BinaryFormat::Xcoff {
        file.add_section(Vec::new(), b".info".to_vec(), SectionKind::Debug)
    } else {
        file.add_section(
            file.segment_name(StandardSegment::Debug).to_vec(),
            section_name.into_bytes(),
            SectionKind::Debug,
        )
    };
    match file.format() {
        BinaryFormat::Coff => {
            file.section_mut(section).flags =
                SectionFlags::Coff { characteristics: pe::IMAGE_SCN_LNK_REMOVE };
        }
        BinaryFormat::Elf => {
            file.section_mut(section).flags =
                SectionFlags::Elf { sh_flags: elf::SHF_EXCLUDE as u64 };
        }
        BinaryFormat::Xcoff => {
            // AIX system linker may aborts if it meets a valid XCOFF file in archive with no .text, no .data and no .bss.
            file.add_section(Vec::new(), b".text".to_vec(), SectionKind::Text);
            file.section_mut(section).flags =
                SectionFlags::Xcoff { s_flags: xcoff::STYP_INFO as u32 };
            // Encode string stored in .info section of XCOFF.
            // FIXME: The length of data here is not guaranteed to fit in a u32.
            // We may have to split the data into multiple pieces in order to
            // store in .info section.
            let len: u32 = data.len().try_into().unwrap();
            let offset = file.append_section_data(section, &len.to_be_bytes(), 1);
            // Add a symbol referring to the data in .info section.
            file.add_symbol(Symbol {
                name: AIX_METADATA_SYMBOL_NAME.into(),
                value: offset + 4,
                size: 0,
                kind: SymbolKind::Unknown,
                scope: SymbolScope::Compilation,
                weak: false,
                section: SymbolSection::Section(section),
                flags: SymbolFlags::Xcoff {
                    n_sclass: xcoff::C_INFO,
                    x_smtyp: xcoff::C_HIDEXT,
                    x_smclas: xcoff::C_HIDEXT,
                    containing_csect: None,
                },
            });
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
    let mut packed_metadata = rustc_metadata::METADATA_HEADER.to_vec();
    packed_metadata.write_all(&(metadata.raw_data().len() as u64).to_le_bytes()).unwrap();
    packed_metadata.extend(metadata.raw_data());

    let Some(mut file) = create_object_file(sess) else {
        if sess.target.is_like_wasm {
            return create_metadata_file_for_wasm(sess, &packed_metadata, ".rustc");
        }
        return packed_metadata.to_vec();
    };
    if file.format() == BinaryFormat::Xcoff {
        return create_compressed_metadata_file_for_xcoff(file, &packed_metadata, symbol_name);
    }
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
    let offset = file.append_section_data(section, &packed_metadata, 1);

    // For MachO and probably PE this is necessary to prevent the linker from throwing away the
    // .rustc section. For ELF this isn't necessary, but it also doesn't harm.
    file.add_symbol(Symbol {
        name: symbol_name.as_bytes().to_vec(),
        value: offset,
        size: packed_metadata.len() as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });

    file.write().unwrap()
}

/// * Xcoff - On AIX, custom sections are merged into predefined sections,
///   so custom .rustc section is not preserved during linking.
///   For this reason, we store metadata in predefined .info section, and
///   define a symbol to reference the metadata. To preserve metadata during
///   linking on AIX, we have to
///   1. Create an empty .text section, a empty .data section.
///   2. Define an empty symbol named `symbol_name` inside .data section.
///   3. Define an symbol named `AIX_METADATA_SYMBOL_NAME` referencing
///      data inside .info section.
///   From XCOFF's view, (2) creates a csect entry in the symbol table, the
///   symbol created by (3) is a info symbol for the preceding csect. Thus
///   two symbols are preserved during linking and we can use the second symbol
///   to reference the metadata.
pub fn create_compressed_metadata_file_for_xcoff(
    mut file: write::Object<'_>,
    data: &[u8],
    symbol_name: &str,
) -> Vec<u8> {
    assert!(file.format() == BinaryFormat::Xcoff);
    // AIX system linker may aborts if it meets a valid XCOFF file in archive with no .text, no .data and no .bss.
    file.add_section(Vec::new(), b".text".to_vec(), SectionKind::Text);
    let data_section = file.add_section(Vec::new(), b".data".to_vec(), SectionKind::Data);
    let section = file.add_section(Vec::new(), b".info".to_vec(), SectionKind::Debug);
    file.add_file_symbol("lib.rmeta".into());
    file.section_mut(section).flags = SectionFlags::Xcoff { s_flags: xcoff::STYP_INFO as u32 };
    // Add a global symbol to data_section.
    file.add_symbol(Symbol {
        name: symbol_name.as_bytes().into(),
        value: 0,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Dynamic,
        weak: true,
        section: SymbolSection::Section(data_section),
        flags: SymbolFlags::None,
    });
    let len: u32 = data.len().try_into().unwrap();
    let offset = file.append_section_data(section, &len.to_be_bytes(), 1);
    // Add a symbol referring to the rustc metadata.
    file.add_symbol(Symbol {
        name: AIX_METADATA_SYMBOL_NAME.into(),
        value: offset + 4, // The metadata is preceded by a 4-byte length field.
        size: 0,
        kind: SymbolKind::Unknown,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::Xcoff {
            n_sclass: xcoff::C_INFO,
            x_smtyp: xcoff::C_HIDEXT,
            x_smclas: xcoff::C_HIDEXT,
            containing_csect: None,
        },
    });
    file.append_section_data(section, data, 1);
    file.write().unwrap()
}

/// Creates a simple WebAssembly object file, which is itself a wasm module,
/// that contains a custom section of the name `section_name` with contents
/// `data`.
///
/// NB: the `object` crate does not yet have support for writing the wasm
/// object file format. In lieu of that the `wasm-encoder` crate is used to
/// build a wasm file by hand.
///
/// The wasm object file format is defined at
/// <https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md>
/// and mainly consists of a `linking` custom section. In this case the custom
/// section there is empty except for a version marker indicating what format
/// it's in.
///
/// The main purpose of this is to contain a custom section with `section_name`,
/// which is then appended after `linking`.
///
/// As a further detail the object needs to have a 64-bit memory if `wasm64` is
/// the target or otherwise it's interpreted as a 32-bit object which is
/// incompatible with 64-bit ones.
pub fn create_metadata_file_for_wasm(sess: &Session, data: &[u8], section_name: &str) -> Vec<u8> {
    assert!(sess.target.is_like_wasm);
    let mut module = wasm_encoder::Module::new();
    let mut imports = wasm_encoder::ImportSection::new();

    if sess.target.pointer_width == 64 {
        imports.import("env", "__linear_memory", wasm_encoder::MemoryType {
            minimum: 0,
            maximum: None,
            memory64: true,
            shared: false,
            page_size_log2: None,
        });
    }

    if imports.len() > 0 {
        module.section(&imports);
    }
    module.section(&wasm_encoder::CustomSection {
        name: "linking".into(),
        data: Cow::Borrowed(&[2]),
    });
    module.section(&wasm_encoder::CustomSection { name: section_name.into(), data: data.into() });
    module.finish()
}
