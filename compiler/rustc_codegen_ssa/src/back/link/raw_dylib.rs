use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustc_abi::Endian;
use rustc_data_structures::base_n::{CASE_INSENSITIVE, ToBaseN};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_hashes::Hash128;
use rustc_session::Session;
use rustc_session::cstore::DllImport;
use rustc_session::utils::NativeLibKind;
use rustc_span::Symbol;

use crate::back::archive::ImportLibraryItem;
use crate::back::link::ArchiveBuilderBuilder;
use crate::errors::ErrorCreatingImportLibrary;
use crate::{NativeLib, common, errors};

/// Extract all symbols defined in raw-dylib libraries, collated by library name.
///
/// If we have multiple extern blocks that specify symbols defined in the same raw-dylib library,
/// then the CodegenResults value contains one NativeLib instance for each block. However, the
/// linker appears to expect only a single import library for each library used, so we need to
/// collate the symbols together by library name before generating the import libraries.
fn collate_raw_dylibs_windows<'a>(
    sess: &Session,
    used_libraries: impl IntoIterator<Item = &'a NativeLib>,
) -> Vec<(String, Vec<DllImport>)> {
    // Use index maps to preserve original order of imports and libraries.
    let mut dylib_table = FxIndexMap::<String, FxIndexMap<Symbol, &DllImport>>::default();

    for lib in used_libraries {
        if lib.kind == NativeLibKind::RawDylib {
            let ext = if lib.verbatim { "" } else { ".dll" };
            let name = format!("{}{}", lib.name, ext);
            let imports = dylib_table.entry(name.clone()).or_default();
            for import in &lib.dll_imports {
                if let Some(old_import) = imports.insert(import.name, import) {
                    // FIXME: when we add support for ordinals, figure out if we need to do anything
                    // if we have two DllImport values with the same name but different ordinals.
                    if import.calling_convention != old_import.calling_convention {
                        sess.dcx().emit_err(errors::MultipleExternalFuncDecl {
                            span: import.span,
                            function: import.name,
                            library_name: &name,
                        });
                    }
                }
            }
        }
    }
    sess.dcx().abort_if_errors();
    dylib_table
        .into_iter()
        .map(|(name, imports)| {
            (name, imports.into_iter().map(|(_, import)| import.clone()).collect())
        })
        .collect()
}

pub(super) fn create_raw_dylib_dll_import_libs<'a>(
    sess: &Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    used_libraries: impl IntoIterator<Item = &'a NativeLib>,
    tmpdir: &Path,
    is_direct_dependency: bool,
) -> Vec<PathBuf> {
    collate_raw_dylibs_windows(sess, used_libraries)
        .into_iter()
        .map(|(raw_dylib_name, raw_dylib_imports)| {
            let name_suffix = if is_direct_dependency { "_imports" } else { "_imports_indirect" };
            let output_path = tmpdir.join(format!("{raw_dylib_name}{name_suffix}.lib"));

            let mingw_gnu_toolchain = common::is_mingw_gnu_toolchain(&sess.target);

            let items: Vec<ImportLibraryItem> = raw_dylib_imports
                .iter()
                .map(|import: &DllImport| {
                    if sess.target.arch == "x86" {
                        ImportLibraryItem {
                            name: common::i686_decorated_name(
                                import,
                                mingw_gnu_toolchain,
                                false,
                                false,
                            ),
                            ordinal: import.ordinal(),
                            symbol_name: import.is_missing_decorations().then(|| {
                                common::i686_decorated_name(
                                    import,
                                    mingw_gnu_toolchain,
                                    false,
                                    true,
                                )
                            }),
                            is_data: !import.is_fn,
                        }
                    } else {
                        ImportLibraryItem {
                            name: import.name.to_string(),
                            ordinal: import.ordinal(),
                            symbol_name: None,
                            is_data: !import.is_fn,
                        }
                    }
                })
                .collect();

            archive_builder_builder.create_dll_import_lib(
                sess,
                &raw_dylib_name,
                items,
                &output_path,
            );

            output_path
        })
        .collect()
}

/// Extract all symbols defined in raw-dylib libraries, collated by library name.
///
/// If we have multiple extern blocks that specify symbols defined in the same raw-dylib library,
/// then the CodegenResults value contains one NativeLib instance for each block. However, the
/// linker appears to expect only a single import library for each library used, so we need to
/// collate the symbols together by library name before generating the import libraries.
fn collate_raw_dylibs_elf<'a>(
    sess: &Session,
    used_libraries: impl IntoIterator<Item = &'a NativeLib>,
) -> Vec<(String, Vec<DllImport>)> {
    // Use index maps to preserve original order of imports and libraries.
    let mut dylib_table = FxIndexMap::<String, FxIndexMap<Symbol, &DllImport>>::default();

    for lib in used_libraries {
        if lib.kind == NativeLibKind::RawDylib {
            let filename = if lib.verbatim {
                lib.name.as_str().to_owned()
            } else {
                let ext = sess.target.dll_suffix.as_ref();
                let prefix = sess.target.dll_prefix.as_ref();
                format!("{prefix}{}{ext}", lib.name)
            };

            let imports = dylib_table.entry(filename.clone()).or_default();
            for import in &lib.dll_imports {
                imports.insert(import.name, import);
            }
        }
    }
    sess.dcx().abort_if_errors();
    dylib_table
        .into_iter()
        .map(|(name, imports)| {
            (name, imports.into_iter().map(|(_, import)| import.clone()).collect())
        })
        .collect()
}

pub(super) fn create_raw_dylib_elf_stub_shared_objects<'a>(
    sess: &Session,
    used_libraries: impl IntoIterator<Item = &'a NativeLib>,
    raw_dylib_so_dir: &Path,
) -> Vec<String> {
    collate_raw_dylibs_elf(sess, used_libraries)
        .into_iter()
        .map(|(load_filename, raw_dylib_imports)| {
            use std::hash::Hash;

            // `load_filename` is the *target/loader* filename that will end up in NEEDED.
            // Usually this will be something like `libc.so` or `libc.so.6` but with
            // verbatim it might also be an absolute path.
            // To be able to support this properly, we always put this load filename
            // into the SONAME of the library and link it via a temporary file with a random name.
            // This also avoids naming conflicts with non-raw-dylib linkage of the same library.

            let shared_object = create_elf_raw_dylib_stub(sess, &load_filename, &raw_dylib_imports);

            let mut file_name_hasher = StableHasher::new();
            load_filename.hash(&mut file_name_hasher);
            for raw_dylib in raw_dylib_imports {
                raw_dylib.name.as_str().hash(&mut file_name_hasher);
            }

            let library_filename: Hash128 = file_name_hasher.finish();
            let temporary_lib_name = format!(
                "{}{}{}",
                sess.target.dll_prefix,
                library_filename.as_u128().to_base_fixed_len(CASE_INSENSITIVE),
                sess.target.dll_suffix
            );
            let link_path = raw_dylib_so_dir.join(&temporary_lib_name);

            let file = match fs::File::create_new(&link_path) {
                Ok(file) => file,
                Err(error) => sess.dcx().emit_fatal(ErrorCreatingImportLibrary {
                    lib_name: &load_filename,
                    error: error.to_string(),
                }),
            };
            if let Err(error) = BufWriter::new(file).write_all(&shared_object) {
                sess.dcx().emit_fatal(ErrorCreatingImportLibrary {
                    lib_name: &load_filename,
                    error: error.to_string(),
                });
            };

            temporary_lib_name
        })
        .collect()
}

/// Create an ELF .so stub file for raw-dylib.
/// It exports all the provided symbols, but is otherwise empty.
fn create_elf_raw_dylib_stub(sess: &Session, soname: &str, symbols: &[DllImport]) -> Vec<u8> {
    use object::write::elf as write;
    use object::{Architecture, elf};

    let mut stub_buf = Vec::new();

    // Build the stub ELF using the object crate.
    // The high-level portable API does not allow for the fine-grained control we need,
    // so this uses the low-level object::write::elf API.
    // The low-level API consists of two stages: reservation and writing.
    // We first reserve space for all the things in the binary and then write them.
    // It is important that the order of reservation matches the order of writing.
    // The object crate contains many debug asserts that fire if you get this wrong.

    let endianness = match sess.target.options.endian {
        Endian::Little => object::Endianness::Little,
        Endian::Big => object::Endianness::Big,
    };
    let mut stub = write::Writer::new(endianness, true, &mut stub_buf);

    // These initial reservations don't reserve any bytes in the binary yet,
    // they just allocate in the internal data structures.

    // First, we crate the dynamic symbol table. It starts with a null symbol
    // and then all the symbols and their dynamic strings.
    stub.reserve_null_dynamic_symbol_index();

    let dynstrs = symbols
        .iter()
        .map(|sym| {
            stub.reserve_dynamic_symbol_index();
            (sym, stub.add_dynamic_string(sym.name.as_str().as_bytes()))
        })
        .collect::<Vec<_>>();

    let soname = stub.add_dynamic_string(soname.as_bytes());

    // Reserve the sections.
    // We have the minimal sections for a dynamic SO and .text where we point our dummy symbols to.
    stub.reserve_shstrtab_section_index();
    let text_section_name = stub.add_section_name(".text".as_bytes());
    let text_section = stub.reserve_section_index();
    stub.reserve_dynstr_section_index();
    stub.reserve_dynsym_section_index();
    stub.reserve_dynamic_section_index();

    // These reservations now determine the actual layout order of the object file.
    stub.reserve_file_header();
    stub.reserve_shstrtab();
    stub.reserve_section_headers();
    stub.reserve_dynstr();
    stub.reserve_dynsym();
    stub.reserve_dynamic(2); // DT_SONAME, DT_NULL

    // First write the ELF header with the arch information.
    let Some((arch, sub_arch)) = sess.target.object_architecture(&sess.unstable_target_features)
    else {
        sess.dcx().fatal(format!(
            "raw-dylib is not supported for the architecture `{}`",
            sess.target.arch
        ));
    };
    let e_machine = match (arch, sub_arch) {
        (Architecture::Aarch64, None) => elf::EM_AARCH64,
        (Architecture::Aarch64_Ilp32, None) => elf::EM_AARCH64,
        (Architecture::Arm, None) => elf::EM_ARM,
        (Architecture::Avr, None) => elf::EM_AVR,
        (Architecture::Bpf, None) => elf::EM_BPF,
        (Architecture::Csky, None) => elf::EM_CSKY,
        (Architecture::E2K32, None) => elf::EM_MCST_ELBRUS,
        (Architecture::E2K64, None) => elf::EM_MCST_ELBRUS,
        (Architecture::I386, None) => elf::EM_386,
        (Architecture::X86_64, None) => elf::EM_X86_64,
        (Architecture::X86_64_X32, None) => elf::EM_X86_64,
        (Architecture::Hexagon, None) => elf::EM_HEXAGON,
        (Architecture::LoongArch32, None) => elf::EM_LOONGARCH,
        (Architecture::LoongArch64, None) => elf::EM_LOONGARCH,
        (Architecture::M68k, None) => elf::EM_68K,
        (Architecture::Mips, None) => elf::EM_MIPS,
        (Architecture::Mips64, None) => elf::EM_MIPS,
        (Architecture::Mips64_N32, None) => elf::EM_MIPS,
        (Architecture::Msp430, None) => elf::EM_MSP430,
        (Architecture::PowerPc, None) => elf::EM_PPC,
        (Architecture::PowerPc64, None) => elf::EM_PPC64,
        (Architecture::Riscv32, None) => elf::EM_RISCV,
        (Architecture::Riscv64, None) => elf::EM_RISCV,
        (Architecture::S390x, None) => elf::EM_S390,
        (Architecture::Sbf, None) => elf::EM_SBF,
        (Architecture::Sharc, None) => elf::EM_SHARC,
        (Architecture::Sparc, None) => elf::EM_SPARC,
        (Architecture::Sparc32Plus, None) => elf::EM_SPARC32PLUS,
        (Architecture::Sparc64, None) => elf::EM_SPARCV9,
        (Architecture::Xtensa, None) => elf::EM_XTENSA,
        _ => {
            sess.dcx().fatal(format!(
                "raw-dylib is not supported for the architecture `{}`",
                sess.target.arch
            ));
        }
    };

    stub.write_file_header(&write::FileHeader {
        os_abi: crate::back::metadata::elf_os_abi(sess),
        abi_version: 0,
        e_type: object::elf::ET_DYN,
        e_machine,
        e_entry: 0,
        e_flags: crate::back::metadata::elf_e_flags(arch, sess),
    })
    .unwrap();

    // .shstrtab
    stub.write_shstrtab();

    // Section headers
    stub.write_null_section_header();
    stub.write_shstrtab_section_header();
    // Create a dummy .text section for our dummy symbols.
    stub.write_section_header(&write::SectionHeader {
        name: Some(text_section_name),
        sh_type: elf::SHT_PROGBITS,
        sh_flags: 0,
        sh_addr: 0,
        sh_offset: 0,
        sh_size: 0,
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 1,
        sh_entsize: 0,
    });
    stub.write_dynstr_section_header(0);
    stub.write_dynsym_section_header(0, 1);
    stub.write_dynamic_section_header(0);

    // .dynstr
    stub.write_dynstr();

    // .dynsym
    stub.write_null_dynamic_symbol();
    for (_, name) in dynstrs {
        stub.write_dynamic_symbol(&write::Sym {
            name: Some(name),
            st_info: (elf::STB_GLOBAL << 4) | elf::STT_NOTYPE,
            st_other: elf::STV_DEFAULT,
            section: Some(text_section),
            st_shndx: 0, // ignored by object in favor of the `section` field
            st_value: 0,
            st_size: 0,
        });
    }

    // .dynamic
    // the DT_SONAME will be used by the linker to populate DT_NEEDED
    // which the loader uses to find the library.
    // DT_NULL terminates the .dynamic table.
    stub.write_dynamic_string(elf::DT_SONAME, soname);
    stub.write_dynamic(elf::DT_NULL, 0);

    stub_buf
}
