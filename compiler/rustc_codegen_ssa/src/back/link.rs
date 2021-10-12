use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_errors::{ErrorReported, Handler};
use rustc_fs_util::fix_windows_verbatim_for_gcc;
use rustc_hir::def_id::CrateNum;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_session::config::{self, CFGuard, CrateType, DebugInfo, LdImpl, Strip};
use rustc_session::config::{OutputFilenames, OutputType, PrintRequest};
use rustc_session::cstore::DllImport;
use rustc_session::output::{check_file_is_writeable, invalid_output_for_target, out_filename};
use rustc_session::search_paths::PathKind;
use rustc_session::utils::NativeLibKind;
/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.
use rustc_session::{filesearch, Session};
use rustc_span::symbol::Symbol;
use rustc_target::abi::Endian;
use rustc_target::spec::crt_objects::{CrtObjects, CrtObjectsFallback};
use rustc_target::spec::{LinkOutputKind, LinkerFlavor, LldFlavor, SplitDebuginfo};
use rustc_target::spec::{PanicStrategy, RelocModel, RelroLevel, SanitizerSet, Target};

use super::archive::{find_library, ArchiveBuilder};
use super::command::Command;
use super::linker::{self, Linker};
use super::rpath::{self, RPathConfig};
use crate::{
    looks_like_rust_object_file, CodegenResults, CompiledModule, CrateInfo, NativeLib,
    METADATA_FILENAME,
};

use cc::windows_registry;
use object::elf;
use object::write::Object;
use object::{Architecture, BinaryFormat, Endianness, FileFlags, SectionFlags, SectionKind};
use regex::Regex;
use tempfile::Builder as TempFileBuilder;

use std::ffi::OsString;
use std::lazy::OnceCell;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};
use std::{ascii, char, env, fmt, fs, io, mem, str};

pub fn ensure_removed(diag_handler: &Handler, path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            diag_handler.err(&format!("failed to remove {}: {}", path.display(), e));
        }
    }
}

/// Performs the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    outputs: &OutputFilenames,
) -> Result<(), ErrorReported> {
    let _timer = sess.timer("link_binary");
    let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
    for &crate_type in sess.crate_types().iter() {
        // Ignore executable crates if we have -Z no-codegen, as they will error.
        if (sess.opts.debugging_opts.no_codegen || !sess.opts.output_types.should_codegen())
            && !output_metadata
            && crate_type == CrateType::Executable
        {
            continue;
        }

        if invalid_output_for_target(sess, crate_type) {
            bug!(
                "invalid output type `{:?}` for target os `{}`",
                crate_type,
                sess.opts.target_triple
            );
        }

        sess.time("link_binary_check_files_are_writeable", || {
            for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
                check_file_is_writeable(obj, sess);
            }
        });

        if outputs.outputs.should_link() {
            let tmpdir = TempFileBuilder::new()
                .prefix("rustc")
                .tempdir()
                .unwrap_or_else(|err| sess.fatal(&format!("couldn't create a temp dir: {}", err)));
            let path = MaybeTempDir::new(tmpdir, sess.opts.cg.save_temps);
            let out_filename = out_filename(
                sess,
                crate_type,
                outputs,
                &codegen_results.crate_info.local_crate_name.as_str(),
            );
            match crate_type {
                CrateType::Rlib => {
                    let _timer = sess.timer("link_rlib");
                    link_rlib::<B>(
                        sess,
                        codegen_results,
                        RlibFlavor::Normal,
                        &out_filename,
                        &path,
                    )?
                    .build();
                }
                CrateType::Staticlib => {
                    link_staticlib::<B>(sess, codegen_results, &out_filename, &path)?;
                }
                _ => {
                    link_natively::<B>(
                        sess,
                        crate_type,
                        &out_filename,
                        codegen_results,
                        path.as_ref(),
                    );
                }
            }
            if sess.opts.json_artifact_notifications {
                sess.parse_sess.span_diagnostic.emit_artifact_notification(&out_filename, "link");
            }
        }
    }

    // Remove the temporary object file and metadata if we aren't saving temps
    sess.time("link_binary_remove_temps", || {
        if !sess.opts.cg.save_temps {
            let remove_temps_from_module = |module: &CompiledModule| {
                if let Some(ref obj) = module.object {
                    ensure_removed(sess.diagnostic(), obj);
                }

                if let Some(ref obj) = module.dwarf_object {
                    ensure_removed(sess.diagnostic(), obj);
                }
            };

            if sess.opts.output_types.should_link() && !preserve_objects_for_their_debuginfo(sess) {
                for module in &codegen_results.modules {
                    remove_temps_from_module(module);
                }
            }

            if let Some(ref metadata_module) = codegen_results.metadata_module {
                remove_temps_from_module(metadata_module);
            }

            if let Some(ref allocator_module) = codegen_results.allocator_module {
                remove_temps_from_module(allocator_module);
            }
        }
    });

    Ok(())
}

pub fn each_linked_rlib(
    info: &CrateInfo,
    f: &mut dyn FnMut(CrateNum, &Path),
) -> Result<(), String> {
    let crates = info.used_crates.iter();
    let mut fmts = None;
    for (ty, list) in info.dependency_formats.iter() {
        match ty {
            CrateType::Executable
            | CrateType::Staticlib
            | CrateType::Cdylib
            | CrateType::ProcMacro => {
                fmts = Some(list);
                break;
            }
            _ => {}
        }
    }
    let fmts = match fmts {
        Some(f) => f,
        None => return Err("could not find formats for rlibs".to_string()),
    };
    for &cnum in crates {
        match fmts.get(cnum.as_usize() - 1) {
            Some(&Linkage::NotLinked | &Linkage::IncludedFromDylib) => continue,
            Some(_) => {}
            None => return Err("could not find formats for rlibs".to_string()),
        }
        let name = &info.crate_name[&cnum];
        let used_crate_source = &info.used_crate_source[&cnum];
        let path = if let Some((path, _)) = &used_crate_source.rlib {
            path
        } else if used_crate_source.rmeta.is_some() {
            return Err(format!(
                "could not find rlib for: `{}`, found rmeta (metadata) file",
                name
            ));
        } else {
            return Err(format!("could not find rlib for: `{}`", name));
        };
        f(cnum, &path);
    }
    Ok(())
}

/// We use a temp directory here to avoid races between concurrent rustc processes,
/// such as builds in the same directory using the same filename for metadata while
/// building an `.rlib` (stomping over one another), or writing an `.rmeta` into a
/// directory being searched for `extern crate` (observing an incomplete file).
/// The returned path is the temporary file containing the complete metadata.
pub fn emit_metadata(sess: &Session, metadata: &[u8], tmpdir: &MaybeTempDir) -> PathBuf {
    let out_filename = tmpdir.as_ref().join(METADATA_FILENAME);
    let result = fs::write(&out_filename, metadata);

    if let Err(e) = result {
        sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
    }

    out_filename
}

/// Create an 'rlib'.
///
/// An rlib in its current incarnation is essentially a renamed .a file. The rlib primarily contains
/// the object file of the crate, but it also contains all of the object files from native
/// libraries. This is done by unzipping native libraries and inserting all of the contents into
/// this archive.
fn link_rlib<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    flavor: RlibFlavor,
    out_filename: &Path,
    tmpdir: &MaybeTempDir,
) -> Result<B, ErrorReported> {
    info!("preparing rlib to {:?}", out_filename);

    let lib_search_paths = archive_search_paths(sess);

    let mut ab = <B as ArchiveBuilder>::new(sess, out_filename, None);

    for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
        ab.add_file(obj);
    }

    // Note that in this loop we are ignoring the value of `lib.cfg`. That is,
    // we may not be configured to actually include a static library if we're
    // adding it here. That's because later when we consume this rlib we'll
    // decide whether we actually needed the static library or not.
    //
    // To do this "correctly" we'd need to keep track of which libraries added
    // which object files to the archive. We don't do that here, however. The
    // #[link(cfg(..))] feature is unstable, though, and only intended to get
    // liblibc working. In that sense the check below just indicates that if
    // there are any libraries we want to omit object files for at link time we
    // just exclude all custom object files.
    //
    // Eventually if we want to stabilize or flesh out the #[link(cfg(..))]
    // feature then we'll need to figure out how to record what objects were
    // loaded from the libraries found here and then encode that into the
    // metadata of the rlib we're generating somehow.
    for lib in codegen_results.crate_info.used_libraries.iter() {
        match lib.kind {
            NativeLibKind::Static { bundle: None | Some(true), whole_archive: Some(true) }
                if flavor == RlibFlavor::Normal =>
            {
                // Don't allow mixing +bundle with +whole_archive since an rlib may contain
                // multiple native libs, some of which are +whole-archive and some of which are
                // -whole-archive and it isn't clear how we can currently handle such a
                // situation correctly.
                // See https://github.com/rust-lang/rust/issues/88085#issuecomment-901050897
                sess.err(
                    "the linking modifiers `+bundle` and `+whole-archive` are not compatible \
                        with each other when generating rlibs",
                );
            }
            NativeLibKind::Static { bundle: None | Some(true), .. } => {}
            NativeLibKind::Static { bundle: Some(false), .. }
            | NativeLibKind::Dylib { .. }
            | NativeLibKind::Framework { .. }
            | NativeLibKind::RawDylib
            | NativeLibKind::Unspecified => continue,
        }
        if let Some(name) = lib.name {
            let location =
                find_library(name, lib.verbatim.unwrap_or(false), &lib_search_paths, sess);
            ab.add_archive(&location, |_| false).unwrap_or_else(|e| {
                sess.fatal(&format!(
                    "failed to add native library {}: {}",
                    location.to_string_lossy(),
                    e
                ));
            });
        }
    }

    for (raw_dylib_name, raw_dylib_imports) in
        collate_raw_dylibs(sess, &codegen_results.crate_info.used_libraries)?
    {
        ab.inject_dll_import_lib(&raw_dylib_name, &raw_dylib_imports, tmpdir);
    }

    // After adding all files to the archive, we need to update the
    // symbol table of the archive.
    ab.update_symbols();

    // Note that it is important that we add all of our non-object "magical
    // files" *after* all of the object files in the archive. The reason for
    // this is as follows:
    //
    // * When performing LTO, this archive will be modified to remove
    //   objects from above. The reason for this is described below.
    //
    // * When the system linker looks at an archive, it will attempt to
    //   determine the architecture of the archive in order to see whether its
    //   linkable.
    //
    //   The algorithm for this detection is: iterate over the files in the
    //   archive. Skip magical SYMDEF names. Interpret the first file as an
    //   object file. Read architecture from the object file.
    //
    // * As one can probably see, if "metadata" and "foo.bc" were placed
    //   before all of the objects, then the architecture of this archive would
    //   not be correctly inferred once 'foo.o' is removed.
    //
    // Basically, all this means is that this code should not move above the
    // code above.
    match flavor {
        RlibFlavor::Normal => {
            // metadata in rlib files is wrapped in a "dummy" object file for
            // the target platform so the rlib can be processed entirely by
            // normal linkers for the platform.
            let metadata = create_metadata_file(sess, codegen_results.metadata.raw_data());
            ab.add_file(&emit_metadata(sess, &metadata, tmpdir));

            // After adding all files to the archive, we need to update the
            // symbol table of the archive. This currently dies on macOS (see
            // #11162), and isn't necessary there anyway
            if !sess.target.is_like_osx {
                ab.update_symbols();
            }
        }

        RlibFlavor::StaticlibBase => {
            let obj = codegen_results.allocator_module.as_ref().and_then(|m| m.object.as_ref());
            if let Some(obj) = obj {
                ab.add_file(obj);
            }
        }
    }
    return Ok(ab);

    // For rlibs we "pack" rustc metadata into a dummy object file. When rustc
    // creates a dylib crate type it will pass `--whole-archive` (or the
    // platform equivalent) to include all object files from an rlib into the
    // final dylib itself. This causes linkers to iterate and try to include all
    // files located in an archive, so if metadata is stored in an archive then
    // it needs to be of a form that the linker will be able to process.
    //
    // Note, though, that we don't actually want this metadata to show up in any
    // final output of the compiler. Instead this is purely for rustc's own
    // metadata tracking purposes.
    //
    // With the above in mind, each "flavor" of object format gets special
    // handling here depending on the target:
    //
    // * MachO - macos-like targets will insert the metadata into a section that
    //   is sort of fake dwarf debug info. Inspecting the source of the macos
    //   linker this causes these sections to be skipped automatically because
    //   it's not in an allowlist of otherwise well known dwarf section names to
    //   go into the final artifact.
    //
    // * WebAssembly - we actually don't have any container format for this
    //   target. WebAssembly doesn't support the `dylib` crate type anyway so
    //   there's no need for us to support this at this time. Consequently the
    //   metadata bytes are simply stored as-is into an rlib.
    //
    // * COFF - Windows-like targets create an object with a section that has
    //   the `IMAGE_SCN_LNK_REMOVE` flag set which ensures that if the linker
    //   ever sees the section it doesn't process it and it's removed.
    //
    // * ELF - All other targets are similar to Windows in that there's a
    //   `SHF_EXCLUDE` flag we can set on sections in an object file to get
    //   automatically removed from the final output.
    //
    // Note that this metdata format is kept in sync with
    // `rustc_codegen_ssa/src/back/metadata.rs`.
    fn create_metadata_file(sess: &Session, metadata: &[u8]) -> Vec<u8> {
        let endianness = match sess.target.options.endian {
            Endian::Little => Endianness::Little,
            Endian::Big => Endianness::Big,
        };
        let architecture = match &sess.target.arch[..] {
            "arm" => Architecture::Arm,
            "aarch64" => Architecture::Aarch64,
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
            _ => return metadata.to_vec(),
        };

        if sess.target.is_like_osx {
            let mut file = Object::new(BinaryFormat::MachO, architecture, endianness);

            let section =
                file.add_section(b"__DWARF".to_vec(), b".rmeta".to_vec(), SectionKind::Debug);
            file.append_section_data(section, metadata, 1);
            file.write().unwrap()
        } else if sess.target.is_like_windows {
            const IMAGE_SCN_LNK_REMOVE: u32 = 0;
            let mut file = Object::new(BinaryFormat::Coff, architecture, endianness);

            let section = file.add_section(Vec::new(), b".rmeta".to_vec(), SectionKind::Debug);
            file.section_mut(section).flags =
                SectionFlags::Coff { characteristics: IMAGE_SCN_LNK_REMOVE };
            file.append_section_data(section, metadata, 1);
            file.write().unwrap()
        } else {
            const SHF_EXCLUDE: u64 = 0x80000000;
            let mut file = Object::new(BinaryFormat::Elf, architecture, endianness);

            match &sess.target.arch[..] {
                // copied from `mipsel-linux-gnu-gcc foo.c -c` and
                // inspecting the resulting `e_flags` field.
                "mips" => {
                    let e_flags = elf::EF_MIPS_ARCH_32R2 | elf::EF_MIPS_CPIC | elf::EF_MIPS_PIC;
                    file.flags = FileFlags::Elf { e_flags };
                }
                // copied from `mips64el-linux-gnuabi64-gcc foo.c -c`
                "mips64" => {
                    let e_flags = elf::EF_MIPS_ARCH_64R2 | elf::EF_MIPS_CPIC | elf::EF_MIPS_PIC;
                    file.flags = FileFlags::Elf { e_flags };
                }

                // copied from `riscv64-linux-gnu-gcc foo.c -c`, note though
                // that the `+d` target feature represents whether the double
                // float abi is enabled.
                "riscv64" if sess.target.options.features.contains("+d") => {
                    let e_flags = elf::EF_RISCV_RVC | elf::EF_RISCV_FLOAT_ABI_DOUBLE;
                    file.flags = FileFlags::Elf { e_flags };
                }

                _ => {}
            }

            let section = file.add_section(Vec::new(), b".rmeta".to_vec(), SectionKind::Debug);
            file.section_mut(section).flags = SectionFlags::Elf { sh_flags: SHF_EXCLUDE };
            file.append_section_data(section, metadata, 1);
            file.write().unwrap()
        }
    }
}

/// Extract all symbols defined in raw-dylib libraries, collated by library name.
///
/// If we have multiple extern blocks that specify symbols defined in the same raw-dylib library,
/// then the CodegenResults value contains one NativeLib instance for each block.  However, the
/// linker appears to expect only a single import library for each library used, so we need to
/// collate the symbols together by library name before generating the import libraries.
fn collate_raw_dylibs(
    sess: &Session,
    used_libraries: &[NativeLib],
) -> Result<Vec<(String, Vec<DllImport>)>, ErrorReported> {
    // Use index maps to preserve original order of imports and libraries.
    let mut dylib_table = FxIndexMap::<String, FxIndexMap<Symbol, &DllImport>>::default();

    for lib in used_libraries {
        if lib.kind == NativeLibKind::RawDylib {
            let ext = if matches!(lib.verbatim, Some(true)) { "" } else { ".dll" };
            let name = format!("{}{}", lib.name.expect("unnamed raw-dylib library"), ext);
            let imports = dylib_table.entry(name.clone()).or_default();
            for import in &lib.dll_imports {
                if let Some(old_import) = imports.insert(import.name, import) {
                    // FIXME: when we add support for ordinals, figure out if we need to do anything
                    // if we have two DllImport values with the same name but different ordinals.
                    if import.calling_convention != old_import.calling_convention {
                        sess.span_err(
                            import.span,
                            &format!(
                                "multiple declarations of external function `{}` from \
                                 library `{}` have different calling conventions",
                                import.name, name,
                            ),
                        );
                    }
                }
            }
        }
    }
    sess.compile_status()?;
    Ok(dylib_table
        .into_iter()
        .map(|(name, imports)| {
            (name, imports.into_iter().map(|(_, import)| import.clone()).collect())
        })
        .collect())
}

/// Create a static archive.
///
/// This is essentially the same thing as an rlib, but it also involves adding all of the upstream
/// crates' objects into the archive. This will slurp in all of the native libraries of upstream
/// dependencies as well.
///
/// Additionally, there's no way for us to link dynamic libraries, so we warn about all dynamic
/// library dependencies that they're not linked in.
///
/// There's no need to include metadata in a static archive, so ensure to not link in the metadata
/// object file (and also don't prepare the archive with a metadata file).
fn link_staticlib<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    out_filename: &Path,
    tempdir: &MaybeTempDir,
) -> Result<(), ErrorReported> {
    let mut ab =
        link_rlib::<B>(sess, codegen_results, RlibFlavor::StaticlibBase, out_filename, tempdir)?;
    let mut all_native_libs = vec![];

    let res = each_linked_rlib(&codegen_results.crate_info, &mut |cnum, path| {
        let name = &codegen_results.crate_info.crate_name[&cnum];
        let native_libs = &codegen_results.crate_info.native_libraries[&cnum];

        // Here when we include the rlib into our staticlib we need to make a
        // decision whether to include the extra object files along the way.
        // These extra object files come from statically included native
        // libraries, but they may be cfg'd away with #[link(cfg(..))].
        //
        // This unstable feature, though, only needs liblibc to work. The only
        // use case there is where musl is statically included in liblibc.rlib,
        // so if we don't want the included version we just need to skip it. As
        // a result the logic here is that if *any* linked library is cfg'd away
        // we just skip all object files.
        //
        // Clearly this is not sufficient for a general purpose feature, and
        // we'd want to read from the library's metadata to determine which
        // object files come from where and selectively skip them.
        let skip_object_files = native_libs.iter().any(|lib| {
            matches!(lib.kind, NativeLibKind::Static { bundle: None | Some(true), .. })
                && !relevant_lib(sess, lib)
        });

        let lto = are_upstream_rust_objects_already_included(sess)
            && !ignored_for_lto(sess, &codegen_results.crate_info, cnum);

        // Ignoring obj file starting with the crate name
        // as simple comparison is not enough - there
        // might be also an extra name suffix
        let obj_start = name.as_str().to_owned();

        ab.add_archive(path, move |fname: &str| {
            // Ignore metadata files, no matter the name.
            if fname == METADATA_FILENAME {
                return true;
            }

            // Don't include Rust objects if LTO is enabled
            if lto && looks_like_rust_object_file(fname) {
                return true;
            }

            // Otherwise if this is *not* a rust object and we're skipping
            // objects then skip this file
            if skip_object_files && (!fname.starts_with(&obj_start) || !fname.ends_with(".o")) {
                return true;
            }

            // ok, don't skip this
            false
        })
        .unwrap();

        all_native_libs.extend(codegen_results.crate_info.native_libraries[&cnum].iter().cloned());
    });
    if let Err(e) = res {
        sess.fatal(&e);
    }

    ab.update_symbols();
    ab.build();

    if !all_native_libs.is_empty() {
        if sess.opts.prints.contains(&PrintRequest::NativeStaticLibs) {
            print_native_static_libs(sess, &all_native_libs);
        }
    }

    Ok(())
}

fn escape_stdout_stderr_string(s: &[u8]) -> String {
    str::from_utf8(s).map(|s| s.to_owned()).unwrap_or_else(|_| {
        let mut x = "Non-UTF-8 output: ".to_string();
        x.extend(s.iter().flat_map(|&b| ascii::escape_default(b)).map(char::from));
        x
    })
}

const LLVM_DWP_EXECUTABLE: &'static str = "rust-llvm-dwp";

/// Invoke `llvm-dwp` (shipped alongside rustc) to link `dwo` files from Split DWARF into a `dwp`
/// file.
fn link_dwarf_object<'a>(sess: &'a Session, executable_out_filename: &Path) {
    info!("preparing dwp to {}.dwp", executable_out_filename.to_str().unwrap());

    let dwp_out_filename = executable_out_filename.with_extension("dwp");
    let mut cmd = Command::new(LLVM_DWP_EXECUTABLE);
    cmd.arg("-e");
    cmd.arg(executable_out_filename);
    cmd.arg("-o");
    cmd.arg(&dwp_out_filename);

    let mut new_path = sess.get_tools_search_paths(false);
    if let Some(path) = env::var_os("PATH") {
        new_path.extend(env::split_paths(&path));
    }
    let new_path = env::join_paths(new_path).unwrap();
    cmd.env("PATH", new_path);

    info!("{:?}", &cmd);
    match sess.time("run_dwp", || cmd.output()) {
        Ok(prog) if !prog.status.success() => {
            sess.struct_err(&format!(
                "linking dwarf objects with `{}` failed: {}",
                LLVM_DWP_EXECUTABLE, prog.status
            ))
            .note(&format!("{:?}", &cmd))
            .note(&escape_stdout_stderr_string(&prog.stdout))
            .note(&escape_stdout_stderr_string(&prog.stderr))
            .emit();
            info!("linker stderr:\n{}", escape_stdout_stderr_string(&prog.stderr));
            info!("linker stdout:\n{}", escape_stdout_stderr_string(&prog.stdout));
        }
        Ok(_) => {}
        Err(e) => {
            let dwp_not_found = e.kind() == io::ErrorKind::NotFound;
            let mut err = if dwp_not_found {
                sess.struct_err(&format!("linker `{}` not found", LLVM_DWP_EXECUTABLE))
            } else {
                sess.struct_err(&format!("could not exec the linker `{}`", LLVM_DWP_EXECUTABLE))
            };

            err.note(&e.to_string());

            if !dwp_not_found {
                err.note(&format!("{:?}", &cmd));
            }

            err.emit();
        }
    }
}

/// Create a dynamic library or executable.
///
/// This will invoke the system linker/cc to create the resulting file. This links to all upstream
/// files as well.
fn link_natively<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    crate_type: CrateType,
    out_filename: &Path,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
) {
    info!("preparing {:?} to {:?}", crate_type, out_filename);
    let (linker_path, flavor) = linker_and_flavor(sess);
    let mut cmd = linker_with_args::<B>(
        &linker_path,
        flavor,
        sess,
        crate_type,
        tmpdir,
        out_filename,
        codegen_results,
    );

    linker::disable_localization(&mut cmd);

    for &(ref k, ref v) in &sess.target.link_env {
        cmd.env(k, v);
    }
    for k in &sess.target.link_env_remove {
        cmd.env_remove(k);
    }

    if sess.opts.debugging_opts.print_link_args {
        println!("{:?}", &cmd);
    }

    // May have not found libraries in the right formats.
    sess.abort_if_errors();

    // Invoke the system linker
    info!("{:?}", &cmd);
    let retry_on_segfault = env::var("RUSTC_RETRY_LINKER_ON_SEGFAULT").is_ok();
    let unknown_arg_regex =
        Regex::new(r"(unknown|unrecognized) (command line )?(option|argument)").unwrap();
    let mut prog;
    let mut i = 0;
    loop {
        i += 1;
        prog = sess.time("run_linker", || exec_linker(sess, &cmd, out_filename, tmpdir));
        let output = match prog {
            Ok(ref output) => output,
            Err(_) => break,
        };
        if output.status.success() {
            break;
        }
        let mut out = output.stderr.clone();
        out.extend(&output.stdout);
        let out = String::from_utf8_lossy(&out);

        // Check to see if the link failed with an error message that indicates it
        // doesn't recognize the -no-pie option. If so, reperform the link step
        // without it. This is safe because if the linker doesn't support -no-pie
        // then it should not default to linking executables as pie. Different
        // versions of gcc seem to use different quotes in the error message so
        // don't check for them.
        if sess.target.linker_is_gnu
            && flavor != LinkerFlavor::Ld
            && unknown_arg_regex.is_match(&out)
            && out.contains("-no-pie")
            && cmd.get_args().iter().any(|e| e.to_string_lossy() == "-no-pie")
        {
            info!("linker output: {:?}", out);
            warn!("Linker does not support -no-pie command line option. Retrying without.");
            for arg in cmd.take_args() {
                if arg.to_string_lossy() != "-no-pie" {
                    cmd.arg(arg);
                }
            }
            info!("{:?}", &cmd);
            continue;
        }

        // Detect '-static-pie' used with an older version of gcc or clang not supporting it.
        // Fallback from '-static-pie' to '-static' in that case.
        if sess.target.linker_is_gnu
            && flavor != LinkerFlavor::Ld
            && unknown_arg_regex.is_match(&out)
            && (out.contains("-static-pie") || out.contains("--no-dynamic-linker"))
            && cmd.get_args().iter().any(|e| e.to_string_lossy() == "-static-pie")
        {
            info!("linker output: {:?}", out);
            warn!(
                "Linker does not support -static-pie command line option. Retrying with -static instead."
            );
            // Mirror `add_(pre,post)_link_objects` to replace CRT objects.
            let self_contained = crt_objects_fallback(sess, crate_type);
            let opts = &sess.target;
            let pre_objects = if self_contained {
                &opts.pre_link_objects_fallback
            } else {
                &opts.pre_link_objects
            };
            let post_objects = if self_contained {
                &opts.post_link_objects_fallback
            } else {
                &opts.post_link_objects
            };
            let get_objects = |objects: &CrtObjects, kind| {
                objects
                    .get(&kind)
                    .iter()
                    .copied()
                    .flatten()
                    .map(|obj| get_object_file_path(sess, obj, self_contained).into_os_string())
                    .collect::<Vec<_>>()
            };
            let pre_objects_static_pie = get_objects(pre_objects, LinkOutputKind::StaticPicExe);
            let post_objects_static_pie = get_objects(post_objects, LinkOutputKind::StaticPicExe);
            let mut pre_objects_static = get_objects(pre_objects, LinkOutputKind::StaticNoPicExe);
            let mut post_objects_static = get_objects(post_objects, LinkOutputKind::StaticNoPicExe);
            // Assume that we know insertion positions for the replacement arguments from replaced
            // arguments, which is true for all supported targets.
            assert!(pre_objects_static.is_empty() || !pre_objects_static_pie.is_empty());
            assert!(post_objects_static.is_empty() || !post_objects_static_pie.is_empty());
            for arg in cmd.take_args() {
                if arg.to_string_lossy() == "-static-pie" {
                    // Replace the output kind.
                    cmd.arg("-static");
                } else if pre_objects_static_pie.contains(&arg) {
                    // Replace the pre-link objects (replace the first and remove the rest).
                    cmd.args(mem::take(&mut pre_objects_static));
                } else if post_objects_static_pie.contains(&arg) {
                    // Replace the post-link objects (replace the first and remove the rest).
                    cmd.args(mem::take(&mut post_objects_static));
                } else {
                    cmd.arg(arg);
                }
            }
            info!("{:?}", &cmd);
            continue;
        }

        // Here's a terribly awful hack that really shouldn't be present in any
        // compiler. Here an environment variable is supported to automatically
        // retry the linker invocation if the linker looks like it segfaulted.
        //
        // Gee that seems odd, normally segfaults are things we want to know
        // about!  Unfortunately though in rust-lang/rust#38878 we're
        // experiencing the linker segfaulting on Travis quite a bit which is
        // causing quite a bit of pain to land PRs when they spuriously fail
        // due to a segfault.
        //
        // The issue #38878 has some more debugging information on it as well,
        // but this unfortunately looks like it's just a race condition in
        // macOS's linker with some thread pool working in the background. It
        // seems that no one currently knows a fix for this so in the meantime
        // we're left with this...
        if !retry_on_segfault || i > 3 {
            break;
        }
        let msg_segv = "clang: error: unable to execute command: Segmentation fault: 11";
        let msg_bus = "clang: error: unable to execute command: Bus error: 10";
        if out.contains(msg_segv) || out.contains(msg_bus) {
            warn!(
                ?cmd, %out,
                "looks like the linker segfaulted when we tried to call it, \
                 automatically retrying again",
            );
            continue;
        }

        if is_illegal_instruction(&output.status) {
            warn!(
                ?cmd, %out, status = %output.status,
                "looks like the linker hit an illegal instruction when we \
                 tried to call it, automatically retrying again.",
            );
            continue;
        }

        #[cfg(unix)]
        fn is_illegal_instruction(status: &ExitStatus) -> bool {
            use std::os::unix::prelude::*;
            status.signal() == Some(libc::SIGILL)
        }

        #[cfg(not(unix))]
        fn is_illegal_instruction(_status: &ExitStatus) -> bool {
            false
        }
    }

    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                let escaped_output = escape_stdout_stderr_string(&output);
                let mut err = sess.struct_err(&format!(
                    "linking with `{}` failed: {}",
                    linker_path.display(),
                    prog.status
                ));
                err.note(&format!("{:?}", &cmd)).note(&escaped_output);
                if escaped_output.contains("undefined reference to") {
                    err.help(
                        "some `extern` functions couldn't be found; some native libraries may \
                         need to be installed or have their path specified",
                    );
                    err.note("use the `-l` flag to specify native libraries to link");
                    err.note("use the `cargo:rustc-link-lib` directive to specify the native \
                              libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)");
                }
                err.emit();

                // If MSVC's `link.exe` was expected but the return code
                // is not a Microsoft LNK error then suggest a way to fix or
                // install the Visual Studio build tools.
                if let Some(code) = prog.status.code() {
                    if sess.target.is_like_msvc
                        && flavor == LinkerFlavor::Msvc
                        // Respect the command line override
                        && sess.opts.cg.linker.is_none()
                        // Match exactly "link.exe"
                        && linker_path.to_str() == Some("link.exe")
                        // All Microsoft `link.exe` linking error codes are
                        // four digit numbers in the range 1000 to 9999 inclusive
                        && (code < 1000 || code > 9999)
                    {
                        let is_vs_installed = windows_registry::find_vs_version().is_ok();
                        let has_linker = windows_registry::find_tool(
                            &sess.opts.target_triple.triple(),
                            "link.exe",
                        )
                        .is_some();

                        sess.note_without_error("`link.exe` returned an unexpected error");
                        if is_vs_installed && has_linker {
                            // the linker is broken
                            sess.note_without_error(
                                "the Visual Studio build tools may need to be repaired \
                                using the Visual Studio installer",
                            );
                            sess.note_without_error(
                                "or a necessary component may be missing from the \
                                \"C++ build tools\" workload",
                            );
                        } else if is_vs_installed {
                            // the linker is not installed
                            sess.note_without_error(
                                "in the Visual Studio installer, ensure the \
                                \"C++ build tools\" workload is selected",
                            );
                        } else {
                            // visual studio is not installed
                            sess.note_without_error(
                                "you may need to install Visual Studio build tools with the \
                                \"C++ build tools\" workload",
                            );
                        }
                    }
                }

                sess.abort_if_errors();
            }
            info!("linker stderr:\n{}", escape_stdout_stderr_string(&prog.stderr));
            info!("linker stdout:\n{}", escape_stdout_stderr_string(&prog.stdout));
        }
        Err(e) => {
            let linker_not_found = e.kind() == io::ErrorKind::NotFound;

            let mut linker_error = {
                if linker_not_found {
                    sess.struct_err(&format!("linker `{}` not found", linker_path.display()))
                } else {
                    sess.struct_err(&format!(
                        "could not exec the linker `{}`",
                        linker_path.display()
                    ))
                }
            };

            linker_error.note(&e.to_string());

            if !linker_not_found {
                linker_error.note(&format!("{:?}", &cmd));
            }

            linker_error.emit();

            if sess.target.is_like_msvc && linker_not_found {
                sess.note_without_error(
                    "the msvc targets depend on the msvc linker \
                     but `link.exe` was not found",
                );
                sess.note_without_error(
                    "please ensure that VS 2013, VS 2015, VS 2017 or VS 2019 \
                     was installed with the Visual C++ option",
                );
            }
            sess.abort_if_errors();
        }
    }

    match sess.split_debuginfo() {
        // If split debug information is disabled or located in individual files
        // there's nothing to do here.
        SplitDebuginfo::Off | SplitDebuginfo::Unpacked => {}

        // If packed split-debuginfo is requested, but the final compilation
        // doesn't actually have any debug information, then we skip this step.
        SplitDebuginfo::Packed if sess.opts.debuginfo == DebugInfo::None => {}

        // On macOS the external `dsymutil` tool is used to create the packed
        // debug information. Note that this will read debug information from
        // the objects on the filesystem which we'll clean up later.
        SplitDebuginfo::Packed if sess.target.is_like_osx => {
            let prog = Command::new("dsymutil").arg(out_filename).output();
            match prog {
                Ok(prog) => {
                    if !prog.status.success() {
                        let mut output = prog.stderr.clone();
                        output.extend_from_slice(&prog.stdout);
                        sess.struct_warn(&format!(
                            "processing debug info with `dsymutil` failed: {}",
                            prog.status
                        ))
                        .note(&escape_string(&output))
                        .emit();
                    }
                }
                Err(e) => sess.fatal(&format!("unable to run `dsymutil`: {}", e)),
            }
        }

        // On MSVC packed debug information is produced by the linker itself so
        // there's no need to do anything else here.
        SplitDebuginfo::Packed if sess.target.is_like_msvc => {}

        // ... and otherwise we're processing a `*.dwp` packed dwarf file.
        SplitDebuginfo::Packed => link_dwarf_object(sess, &out_filename),
    }

    if sess.target.is_like_osx {
        match sess.opts.debugging_opts.strip {
            Strip::Debuginfo => strip_symbols_in_osx(sess, &out_filename, Some("-S")),
            Strip::Symbols => strip_symbols_in_osx(sess, &out_filename, None),
            Strip::None => {}
        }
    }
}

fn strip_symbols_in_osx<'a>(sess: &'a Session, out_filename: &Path, option: Option<&str>) {
    let mut cmd = Command::new("strip");
    if let Some(option) = option {
        cmd.arg(option);
    }
    let prog = cmd.arg(out_filename).output();
    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                sess.struct_warn(&format!(
                    "stripping debug info with `strip` failed: {}",
                    prog.status
                ))
                .note(&escape_string(&output))
                .emit();
            }
        }
        Err(e) => sess.fatal(&format!("unable to run `strip`: {}", e)),
    }
}

fn escape_string(s: &[u8]) -> String {
    str::from_utf8(s).map(|s| s.to_owned()).unwrap_or_else(|_| {
        let mut x = "Non-UTF-8 output: ".to_string();
        x.extend(s.iter().flat_map(|&b| ascii::escape_default(b)).map(char::from));
        x
    })
}

fn add_sanitizer_libraries(sess: &Session, crate_type: CrateType, linker: &mut dyn Linker) {
    // On macOS the runtimes are distributed as dylibs which should be linked to
    // both executables and dynamic shared objects. Everywhere else the runtimes
    // are currently distributed as static liraries which should be linked to
    // executables only.
    let needs_runtime = match crate_type {
        CrateType::Executable => true,
        CrateType::Dylib | CrateType::Cdylib | CrateType::ProcMacro => sess.target.is_like_osx,
        CrateType::Rlib | CrateType::Staticlib => false,
    };

    if !needs_runtime {
        return;
    }

    let sanitizer = sess.opts.debugging_opts.sanitizer;
    if sanitizer.contains(SanitizerSet::ADDRESS) {
        link_sanitizer_runtime(sess, linker, "asan");
    }
    if sanitizer.contains(SanitizerSet::LEAK) {
        link_sanitizer_runtime(sess, linker, "lsan");
    }
    if sanitizer.contains(SanitizerSet::MEMORY) {
        link_sanitizer_runtime(sess, linker, "msan");
    }
    if sanitizer.contains(SanitizerSet::THREAD) {
        link_sanitizer_runtime(sess, linker, "tsan");
    }
    if sanitizer.contains(SanitizerSet::HWADDRESS) {
        link_sanitizer_runtime(sess, linker, "hwasan");
    }
}

fn link_sanitizer_runtime(sess: &Session, linker: &mut dyn Linker, name: &str) {
    fn find_sanitizer_runtime(sess: &Session, filename: &String) -> PathBuf {
        let session_tlib =
            filesearch::make_target_lib_path(&sess.sysroot, sess.opts.target_triple.triple());
        let path = session_tlib.join(&filename);
        if path.exists() {
            return session_tlib;
        } else {
            let default_sysroot = filesearch::get_or_default_sysroot();
            let default_tlib = filesearch::make_target_lib_path(
                &default_sysroot,
                sess.opts.target_triple.triple(),
            );
            return default_tlib;
        }
    }

    let channel = option_env!("CFG_RELEASE_CHANNEL")
        .map(|channel| format!("-{}", channel))
        .unwrap_or_default();

    if sess.target.is_like_osx {
        // On Apple platforms, the sanitizer is always built as a dylib, and
        // LLVM will link to `@rpath/*.dylib`, so we need to specify an
        // rpath to the library as well (the rpath should be absolute, see
        // PR #41352 for details).
        let filename = format!("rustc{}_rt.{}", channel, name);
        let path = find_sanitizer_runtime(&sess, &filename);
        let rpath = path.to_str().expect("non-utf8 component in path");
        linker.args(&["-Wl,-rpath", "-Xlinker", rpath]);
        linker.link_dylib(Symbol::intern(&filename), false, true);
    } else {
        let filename = format!("librustc{}_rt.{}.a", channel, name);
        let path = find_sanitizer_runtime(&sess, &filename).join(&filename);
        linker.link_whole_rlib(&path);
    }
}

/// Returns a boolean indicating whether the specified crate should be ignored
/// during LTO.
///
/// Crates ignored during LTO are not lumped together in the "massive object
/// file" that we create and are linked in their normal rlib states. See
/// comments below for what crates do not participate in LTO.
///
/// It's unusual for a crate to not participate in LTO. Typically only
/// compiler-specific and unstable crates have a reason to not participate in
/// LTO.
pub fn ignored_for_lto(sess: &Session, info: &CrateInfo, cnum: CrateNum) -> bool {
    // If our target enables builtin function lowering in LLVM then the
    // crates providing these functions don't participate in LTO (e.g.
    // no_builtins or compiler builtins crates).
    !sess.target.no_builtins
        && (info.compiler_builtins == Some(cnum) || info.is_no_builtins.contains(&cnum))
}

// This functions tries to determine the appropriate linker (and corresponding LinkerFlavor) to use
pub fn linker_and_flavor(sess: &Session) -> (PathBuf, LinkerFlavor) {
    fn infer_from(
        sess: &Session,
        linker: Option<PathBuf>,
        flavor: Option<LinkerFlavor>,
    ) -> Option<(PathBuf, LinkerFlavor)> {
        match (linker, flavor) {
            (Some(linker), Some(flavor)) => Some((linker, flavor)),
            // only the linker flavor is known; use the default linker for the selected flavor
            (None, Some(flavor)) => Some((
                PathBuf::from(match flavor {
                    LinkerFlavor::Em => {
                        if cfg!(windows) {
                            "emcc.bat"
                        } else {
                            "emcc"
                        }
                    }
                    LinkerFlavor::Gcc => {
                        if cfg!(any(target_os = "solaris", target_os = "illumos")) {
                            // On historical Solaris systems, "cc" may have
                            // been Sun Studio, which is not flag-compatible
                            // with "gcc".  This history casts a long shadow,
                            // and many modern illumos distributions today
                            // ship GCC as "gcc" without also making it
                            // available as "cc".
                            "gcc"
                        } else {
                            "cc"
                        }
                    }
                    LinkerFlavor::Ld => "ld",
                    LinkerFlavor::Msvc => "link.exe",
                    LinkerFlavor::Lld(_) => "lld",
                    LinkerFlavor::PtxLinker => "rust-ptx-linker",
                    LinkerFlavor::BpfLinker => "bpf-linker",
                }),
                flavor,
            )),
            (Some(linker), None) => {
                let stem = linker.file_stem().and_then(|stem| stem.to_str()).unwrap_or_else(|| {
                    sess.fatal("couldn't extract file stem from specified linker")
                });

                let flavor = if stem == "emcc" {
                    LinkerFlavor::Em
                } else if stem == "gcc"
                    || stem.ends_with("-gcc")
                    || stem == "clang"
                    || stem.ends_with("-clang")
                {
                    LinkerFlavor::Gcc
                } else if stem == "wasm-ld" || stem.ends_with("-wasm-ld") {
                    LinkerFlavor::Lld(LldFlavor::Wasm)
                } else if stem == "ld" || stem == "ld.lld" || stem.ends_with("-ld") {
                    LinkerFlavor::Ld
                } else if stem == "link" || stem == "lld-link" {
                    LinkerFlavor::Msvc
                } else if stem == "lld" || stem == "rust-lld" {
                    LinkerFlavor::Lld(sess.target.lld_flavor)
                } else {
                    // fall back to the value in the target spec
                    sess.target.linker_flavor
                };

                Some((linker, flavor))
            }
            (None, None) => None,
        }
    }

    // linker and linker flavor specified via command line have precedence over what the target
    // specification specifies
    if let Some(ret) = infer_from(sess, sess.opts.cg.linker.clone(), sess.opts.cg.linker_flavor) {
        return ret;
    }

    if let Some(ret) = infer_from(
        sess,
        sess.target.linker.clone().map(PathBuf::from),
        Some(sess.target.linker_flavor),
    ) {
        return ret;
    }

    bug!("Not enough information provided to determine how to invoke the linker");
}

/// Returns a boolean indicating whether we should preserve the object files on
/// the filesystem for their debug information. This is often useful with
/// split-dwarf like schemes.
fn preserve_objects_for_their_debuginfo(sess: &Session) -> bool {
    // If the objects don't have debuginfo there's nothing to preserve.
    if sess.opts.debuginfo == config::DebugInfo::None {
        return false;
    }

    // If we're only producing artifacts that are archives, no need to preserve
    // the objects as they're losslessly contained inside the archives.
    let output_linked =
        sess.crate_types().iter().any(|&x| x != CrateType::Rlib && x != CrateType::Staticlib);
    if !output_linked {
        return false;
    }

    // "unpacked" split debuginfo means that we leave object files as the
    // debuginfo is found in the original object files themselves
    sess.split_debuginfo() == SplitDebuginfo::Unpacked
}

fn archive_search_paths(sess: &Session) -> Vec<PathBuf> {
    sess.target_filesearch(PathKind::Native).search_path_dirs()
}

#[derive(PartialEq)]
enum RlibFlavor {
    Normal,
    StaticlibBase,
}

fn print_native_static_libs(sess: &Session, all_native_libs: &[NativeLib]) {
    let lib_args: Vec<_> = all_native_libs
        .iter()
        .filter(|l| relevant_lib(sess, l))
        .filter_map(|lib| {
            let name = lib.name?;
            match lib.kind {
                NativeLibKind::Static { bundle: Some(false), .. }
                | NativeLibKind::Dylib { .. }
                | NativeLibKind::Unspecified => {
                    let verbatim = lib.verbatim.unwrap_or(false);
                    if sess.target.is_like_msvc {
                        Some(format!("{}{}", name, if verbatim { "" } else { ".lib" }))
                    } else if sess.target.linker_is_gnu {
                        Some(format!("-l{}{}", if verbatim { ":" } else { "" }, name))
                    } else {
                        Some(format!("-l{}", name))
                    }
                }
                NativeLibKind::Framework { .. } => {
                    // ld-only syntax, since there are no frameworks in MSVC
                    Some(format!("-framework {}", name))
                }
                // These are included, no need to print them
                NativeLibKind::Static { bundle: None | Some(true), .. }
                | NativeLibKind::RawDylib => None,
            }
        })
        .collect();
    if !lib_args.is_empty() {
        sess.note_without_error(
            "Link against the following native artifacts when linking \
                                 against this static library. The order and any duplication \
                                 can be significant on some platforms.",
        );
        // Prefix for greppability
        sess.note_without_error(&format!("native-static-libs: {}", &lib_args.join(" ")));
    }
}

fn get_object_file_path(sess: &Session, name: &str, self_contained: bool) -> PathBuf {
    let fs = sess.target_filesearch(PathKind::Native);
    let file_path = fs.get_lib_path().join(name);
    if file_path.exists() {
        return file_path;
    }
    // Special directory with objects used only in self-contained linkage mode
    if self_contained {
        let file_path = fs.get_self_contained_lib_path().join(name);
        if file_path.exists() {
            return file_path;
        }
    }
    for search_path in fs.search_paths() {
        let file_path = search_path.dir.join(name);
        if file_path.exists() {
            return file_path;
        }
    }
    PathBuf::from(name)
}

fn exec_linker(
    sess: &Session,
    cmd: &Command,
    out_filename: &Path,
    tmpdir: &Path,
) -> io::Result<Output> {
    // When attempting to spawn the linker we run a risk of blowing out the
    // size limits for spawning a new process with respect to the arguments
    // we pass on the command line.
    //
    // Here we attempt to handle errors from the OS saying "your list of
    // arguments is too big" by reinvoking the linker again with an `@`-file
    // that contains all the arguments. The theory is that this is then
    // accepted on all linkers and the linker will read all its options out of
    // there instead of looking at the command line.
    if !cmd.very_likely_to_exceed_some_spawn_limit() {
        match cmd.command().stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(child) => {
                let output = child.wait_with_output();
                flush_linked_file(&output, out_filename)?;
                return output;
            }
            Err(ref e) if command_line_too_big(e) => {
                info!("command line to linker was too big: {}", e);
            }
            Err(e) => return Err(e),
        }
    }

    info!("falling back to passing arguments to linker via an @-file");
    let mut cmd2 = cmd.clone();
    let mut args = String::new();
    for arg in cmd2.take_args() {
        args.push_str(
            &Escape { arg: arg.to_str().unwrap(), is_like_msvc: sess.target.is_like_msvc }
                .to_string(),
        );
        args.push('\n');
    }
    let file = tmpdir.join("linker-arguments");
    let bytes = if sess.target.is_like_msvc {
        let mut out = Vec::with_capacity((1 + args.len()) * 2);
        // start the stream with a UTF-16 BOM
        for c in std::iter::once(0xFEFF).chain(args.encode_utf16()) {
            // encode in little endian
            out.push(c as u8);
            out.push((c >> 8) as u8);
        }
        out
    } else {
        args.into_bytes()
    };
    fs::write(&file, &bytes)?;
    cmd2.arg(format!("@{}", file.display()));
    info!("invoking linker {:?}", cmd2);
    let output = cmd2.output();
    flush_linked_file(&output, out_filename)?;
    return output;

    #[cfg(not(windows))]
    fn flush_linked_file(_: &io::Result<Output>, _: &Path) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn flush_linked_file(
        command_output: &io::Result<Output>,
        out_filename: &Path,
    ) -> io::Result<()> {
        // On Windows, under high I/O load, output buffers are sometimes not flushed,
        // even long after process exit, causing nasty, non-reproducible output bugs.
        //
        // File::sync_all() calls FlushFileBuffers() down the line, which solves the problem.
        //
        // А full writeup of the original Chrome bug can be found at
        // randomascii.wordpress.com/2018/02/25/compiler-bug-linker-bug-windows-kernel-bug/amp

        if let &Ok(ref out) = command_output {
            if out.status.success() {
                if let Ok(of) = fs::OpenOptions::new().write(true).open(out_filename) {
                    of.sync_all()?;
                }
            }
        }

        Ok(())
    }

    #[cfg(unix)]
    fn command_line_too_big(err: &io::Error) -> bool {
        err.raw_os_error() == Some(::libc::E2BIG)
    }

    #[cfg(windows)]
    fn command_line_too_big(err: &io::Error) -> bool {
        const ERROR_FILENAME_EXCED_RANGE: i32 = 206;
        err.raw_os_error() == Some(ERROR_FILENAME_EXCED_RANGE)
    }

    #[cfg(not(any(unix, windows)))]
    fn command_line_too_big(_: &io::Error) -> bool {
        false
    }

    struct Escape<'a> {
        arg: &'a str,
        is_like_msvc: bool,
    }

    impl<'a> fmt::Display for Escape<'a> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.is_like_msvc {
                // This is "documented" at
                // https://docs.microsoft.com/en-us/cpp/build/reference/at-specify-a-linker-response-file
                //
                // Unfortunately there's not a great specification of the
                // syntax I could find online (at least) but some local
                // testing showed that this seemed sufficient-ish to catch
                // at least a few edge cases.
                write!(f, "\"")?;
                for c in self.arg.chars() {
                    match c {
                        '"' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
                write!(f, "\"")?;
            } else {
                // This is documented at https://linux.die.net/man/1/ld, namely:
                //
                // > Options in file are separated by whitespace. A whitespace
                // > character may be included in an option by surrounding the
                // > entire option in either single or double quotes. Any
                // > character (including a backslash) may be included by
                // > prefixing the character to be included with a backslash.
                //
                // We put an argument on each line, so all we need to do is
                // ensure the line is interpreted as one whole argument.
                for c in self.arg.chars() {
                    match c {
                        '\\' | ' ' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
            }
            Ok(())
        }
    }
}

fn link_output_kind(sess: &Session, crate_type: CrateType) -> LinkOutputKind {
    let kind = match (crate_type, sess.crt_static(Some(crate_type)), sess.relocation_model()) {
        (CrateType::Executable, _, _) if sess.is_wasi_reactor() => LinkOutputKind::WasiReactorExe,
        (CrateType::Executable, false, RelocModel::Pic | RelocModel::Pie) => {
            LinkOutputKind::DynamicPicExe
        }
        (CrateType::Executable, false, _) => LinkOutputKind::DynamicNoPicExe,
        (CrateType::Executable, true, RelocModel::Pic | RelocModel::Pie) => {
            LinkOutputKind::StaticPicExe
        }
        (CrateType::Executable, true, _) => LinkOutputKind::StaticNoPicExe,
        (_, true, _) => LinkOutputKind::StaticDylib,
        (_, false, _) => LinkOutputKind::DynamicDylib,
    };

    // Adjust the output kind to target capabilities.
    let opts = &sess.target;
    let pic_exe_supported = opts.position_independent_executables;
    let static_pic_exe_supported = opts.static_position_independent_executables;
    let static_dylib_supported = opts.crt_static_allows_dylibs;
    match kind {
        LinkOutputKind::DynamicPicExe if !pic_exe_supported => LinkOutputKind::DynamicNoPicExe,
        LinkOutputKind::StaticPicExe if !static_pic_exe_supported => LinkOutputKind::StaticNoPicExe,
        LinkOutputKind::StaticDylib if !static_dylib_supported => LinkOutputKind::DynamicDylib,
        _ => kind,
    }
}

// Returns true if linker is located within sysroot
fn detect_self_contained_mingw(sess: &Session) -> bool {
    let (linker, _) = linker_and_flavor(&sess);
    // Assume `-C linker=rust-lld` as self-contained mode
    if linker == Path::new("rust-lld") {
        return true;
    }
    let linker_with_extension = if cfg!(windows) && linker.extension().is_none() {
        linker.with_extension("exe")
    } else {
        linker
    };
    for dir in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
        let full_path = dir.join(&linker_with_extension);
        // If linker comes from sysroot assume self-contained mode
        if full_path.is_file() && !full_path.starts_with(&sess.sysroot) {
            return false;
        }
    }
    true
}

/// Whether we link to our own CRT objects instead of relying on gcc to pull them.
/// We only provide such support for a very limited number of targets.
fn crt_objects_fallback(sess: &Session, crate_type: CrateType) -> bool {
    if let Some(self_contained) = sess.opts.cg.link_self_contained {
        return self_contained;
    }

    match sess.target.crt_objects_fallback {
        // FIXME: Find a better heuristic for "native musl toolchain is available",
        // based on host and linker path, for example.
        // (https://github.com/rust-lang/rust/pull/71769#issuecomment-626330237).
        Some(CrtObjectsFallback::Musl) => sess.crt_static(Some(crate_type)),
        Some(CrtObjectsFallback::Mingw) => {
            sess.host == sess.target
                && sess.target.vendor != "uwp"
                && detect_self_contained_mingw(&sess)
        }
        // FIXME: Figure out cases in which WASM needs to link with a native toolchain.
        Some(CrtObjectsFallback::Wasm) => true,
        None => false,
    }
}

/// Add pre-link object files defined by the target spec.
fn add_pre_link_objects(
    cmd: &mut dyn Linker,
    sess: &Session,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
) {
    let opts = &sess.target;
    let objects =
        if self_contained { &opts.pre_link_objects_fallback } else { &opts.pre_link_objects };
    for obj in objects.get(&link_output_kind).iter().copied().flatten() {
        cmd.add_object(&get_object_file_path(sess, obj, self_contained));
    }
}

/// Add post-link object files defined by the target spec.
fn add_post_link_objects(
    cmd: &mut dyn Linker,
    sess: &Session,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
) {
    let opts = &sess.target;
    let objects =
        if self_contained { &opts.post_link_objects_fallback } else { &opts.post_link_objects };
    for obj in objects.get(&link_output_kind).iter().copied().flatten() {
        cmd.add_object(&get_object_file_path(sess, obj, self_contained));
    }
}

/// Add arbitrary "pre-link" args defined by the target spec or from command line.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_pre_link_args(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    if let Some(args) = sess.target.pre_link_args.get(&flavor) {
        cmd.args(args);
    }
    cmd.args(&sess.opts.debugging_opts.pre_link_args);
}

/// Add a link script embedded in the target, if applicable.
fn add_link_script(cmd: &mut dyn Linker, sess: &Session, tmpdir: &Path, crate_type: CrateType) {
    match (crate_type, &sess.target.link_script) {
        (CrateType::Cdylib | CrateType::Executable, Some(script)) => {
            if !sess.target.linker_is_gnu {
                sess.fatal("can only use link script when linking with GNU-like linker");
            }

            let file_name = ["rustc", &sess.target.llvm_target, "linkfile.ld"].join("-");

            let path = tmpdir.join(file_name);
            if let Err(e) = fs::write(&path, script) {
                sess.fatal(&format!("failed to write link script to {}: {}", path.display(), e));
            }

            cmd.arg("--script");
            cmd.arg(path);
        }
        _ => {}
    }
}

/// Add arbitrary "user defined" args defined from command line.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_user_defined_link_args(cmd: &mut dyn Linker, sess: &Session) {
    cmd.args(&sess.opts.cg.link_args);
}

/// Add arbitrary "late link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_late_link_args(
    cmd: &mut dyn Linker,
    sess: &Session,
    flavor: LinkerFlavor,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
) {
    let any_dynamic_crate = crate_type == CrateType::Dylib
        || codegen_results.crate_info.dependency_formats.iter().any(|(ty, list)| {
            *ty == crate_type && list.iter().any(|&linkage| linkage == Linkage::Dynamic)
        });
    if any_dynamic_crate {
        if let Some(args) = sess.target.late_link_args_dynamic.get(&flavor) {
            cmd.args(args);
        }
    } else {
        if let Some(args) = sess.target.late_link_args_static.get(&flavor) {
            cmd.args(args);
        }
    }
    if let Some(args) = sess.target.late_link_args.get(&flavor) {
        cmd.args(args);
    }
}

/// Add arbitrary "post-link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_post_link_args(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    if let Some(args) = sess.target.post_link_args.get(&flavor) {
        cmd.args(args);
    }
}

/// Add object files containing code from the current crate.
fn add_local_crate_regular_objects(cmd: &mut dyn Linker, codegen_results: &CodegenResults) {
    for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
        cmd.add_object(obj);
    }
}

/// Add object files for allocator code linked once for the whole crate tree.
fn add_local_crate_allocator_objects(cmd: &mut dyn Linker, codegen_results: &CodegenResults) {
    if let Some(obj) = codegen_results.allocator_module.as_ref().and_then(|m| m.object.as_ref()) {
        cmd.add_object(obj);
    }
}

/// Add object files containing metadata for the current crate.
fn add_local_crate_metadata_objects(
    cmd: &mut dyn Linker,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
) {
    // When linking a dynamic library, we put the metadata into a section of the
    // executable. This metadata is in a separate object file from the main
    // object file, so we link that in here.
    if crate_type == CrateType::Dylib || crate_type == CrateType::ProcMacro {
        if let Some(obj) = codegen_results.metadata_module.as_ref().and_then(|m| m.object.as_ref())
        {
            cmd.add_object(obj);
        }
    }
}

/// Add sysroot and other globally set directories to the directory search list.
fn add_library_search_dirs(cmd: &mut dyn Linker, sess: &Session, self_contained: bool) {
    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = sess.target_filesearch(PathKind::All).get_lib_path();
    cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));

    // Special directory with libraries used only in self-contained linkage mode
    if self_contained {
        let lib_path = sess.target_filesearch(PathKind::All).get_self_contained_lib_path();
        cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));
    }
}

/// Add options making relocation sections in the produced ELF files read-only
/// and suppressing lazy binding.
fn add_relro_args(cmd: &mut dyn Linker, sess: &Session) {
    match sess.opts.debugging_opts.relro_level.unwrap_or(sess.target.relro_level) {
        RelroLevel::Full => cmd.full_relro(),
        RelroLevel::Partial => cmd.partial_relro(),
        RelroLevel::Off => cmd.no_relro(),
        RelroLevel::None => {}
    }
}

/// Add library search paths used at runtime by dynamic linkers.
fn add_rpath_args(
    cmd: &mut dyn Linker,
    sess: &Session,
    codegen_results: &CodegenResults,
    out_filename: &Path,
) {
    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // add_lib_search_paths
    if sess.opts.cg.rpath {
        let libs = codegen_results
            .crate_info
            .used_crates
            .iter()
            .filter_map(|cnum| {
                codegen_results.crate_info.used_crate_source[cnum]
                    .dylib
                    .as_ref()
                    .map(|(path, _)| &**path)
            })
            .collect::<Vec<_>>();
        let mut rpath_config = RPathConfig {
            libs: &*libs,
            out_filename: out_filename.to_path_buf(),
            has_rpath: sess.target.has_rpath,
            is_like_osx: sess.target.is_like_osx,
            linker_is_gnu: sess.target.linker_is_gnu,
        };
        cmd.args(&rpath::get_rpath_flags(&mut rpath_config));
    }
}

/// Produce the linker command line containing linker path and arguments.
///
/// When comments in the function say "order-(in)dependent" they mean order-dependence between
/// options and libraries/object files. For example `--whole-archive` (order-dependent) applies
/// to specific libraries passed after it, and `-o` (output file, order-independent) applies
/// to the linking process as a whole.
/// Order-independent options may still override each other in order-dependent fashion,
/// e.g `--foo=yes --foo=no` may be equivalent to `--foo=no`.
fn linker_with_args<'a, B: ArchiveBuilder<'a>>(
    path: &Path,
    flavor: LinkerFlavor,
    sess: &'a Session,
    crate_type: CrateType,
    tmpdir: &Path,
    out_filename: &Path,
    codegen_results: &CodegenResults,
) -> Command {
    let crt_objects_fallback = crt_objects_fallback(sess, crate_type);
    let cmd = &mut *super::linker::get_linker(
        sess,
        path,
        flavor,
        crt_objects_fallback,
        &codegen_results.crate_info.target_cpu,
    );
    let link_output_kind = link_output_kind(sess, crate_type);

    // ------------ Early order-dependent options ------------

    // If we're building something like a dynamic library then some platforms
    // need to make sure that all symbols are exported correctly from the
    // dynamic library.
    // Must be passed before any libraries to prevent the symbols to export from being thrown away,
    // at least on some platforms (e.g. windows-gnu).
    cmd.export_symbols(
        tmpdir,
        crate_type,
        &codegen_results.crate_info.exported_symbols[&crate_type],
    );

    // Can be used for adding custom CRT objects or overriding order-dependent options above.
    // FIXME: In practice built-in target specs use this for arbitrary order-independent options,
    // introduce a target spec option for order-independent linker options and migrate built-in
    // specs to it.
    add_pre_link_args(cmd, sess, flavor);

    // ------------ Object code and libraries, order-dependent ------------

    // Pre-link CRT objects.
    add_pre_link_objects(cmd, sess, link_output_kind, crt_objects_fallback);

    // Sanitizer libraries.
    add_sanitizer_libraries(sess, crate_type, cmd);

    // Object code from the current crate.
    // Take careful note of the ordering of the arguments we pass to the linker
    // here. Linkers will assume that things on the left depend on things to the
    // right. Things on the right cannot depend on things on the left. This is
    // all formally implemented in terms of resolving symbols (libs on the right
    // resolve unknown symbols of libs on the left, but not vice versa).
    //
    // For this reason, we have organized the arguments we pass to the linker as
    // such:
    //
    // 1. The local object that LLVM just generated
    // 2. Local native libraries
    // 3. Upstream rust libraries
    // 4. Upstream native libraries
    //
    // The rationale behind this ordering is that those items lower down in the
    // list can't depend on items higher up in the list. For example nothing can
    // depend on what we just generated (e.g., that'd be a circular dependency).
    // Upstream rust libraries are not supposed to depend on our local native
    // libraries as that would violate the structure of the DAG, in that
    // scenario they are required to link to them as well in a shared fashion.
    // (The current implementation still doesn't prevent it though, see the FIXME below.)
    //
    // Note that upstream rust libraries may contain native dependencies as
    // well, but they also can't depend on what we just started to add to the
    // link line. And finally upstream native libraries can't depend on anything
    // in this DAG so far because they can only depend on other native libraries
    // and such dependencies are also required to be specified.
    add_local_crate_regular_objects(cmd, codegen_results);
    add_local_crate_metadata_objects(cmd, crate_type, codegen_results);
    add_local_crate_allocator_objects(cmd, codegen_results);

    // Avoid linking to dynamic libraries unless they satisfy some undefined symbols
    // at the point at which they are specified on the command line.
    // Must be passed before any (dynamic) libraries to have effect on them.
    // On Solaris-like systems, `-z ignore` acts as both `--as-needed` and `--gc-sections`
    // so it will ignore unreferenced ELF sections from relocatable objects.
    // For that reason, we put this flag after metadata objects as they would otherwise be removed.
    // FIXME: Support more fine-grained dead code removal on Solaris/illumos
    // and move this option back to the top.
    cmd.add_as_needed();

    // FIXME: Move this below to other native libraries
    // (or alternatively link all native libraries after their respective crates).
    // This change is somewhat breaking in practice due to local static libraries being linked
    // as whole-archive (#85144), so removing whole-archive may be a pre-requisite.
    if sess.opts.debugging_opts.link_native_libraries {
        add_local_native_libraries(cmd, sess, codegen_results);
    }

    // Upstream rust libraries and their nobundle static libraries
    add_upstream_rust_crates::<B>(cmd, sess, codegen_results, crate_type, tmpdir);

    // Upstream dymamic native libraries linked with `#[link]` attributes at and `-l`
    // command line options.
    // If -Zlink-native-libraries=false is set, then the assumption is that an
    // external build system already has the native dependencies defined, and it
    // will provide them to the linker itself.
    if sess.opts.debugging_opts.link_native_libraries {
        add_upstream_native_libraries(cmd, sess, codegen_results);
    }

    // Library linking above uses some global state for things like `-Bstatic`/`-Bdynamic` to make
    // command line shorter, reset it to default here before adding more libraries.
    cmd.reset_per_library_state();

    // FIXME: Built-in target specs occasionally use this for linking system libraries,
    // eliminate all such uses by migrating them to `#[link]` attributes in `lib(std,c,unwind)`
    // and remove the option.
    add_late_link_args(cmd, sess, flavor, crate_type, codegen_results);

    // ------------ Arbitrary order-independent options ------------

    // Add order-independent options determined by rustc from its compiler options,
    // target properties and source code.
    add_order_independent_options(
        cmd,
        sess,
        link_output_kind,
        crt_objects_fallback,
        flavor,
        crate_type,
        codegen_results,
        out_filename,
        tmpdir,
    );

    // Can be used for arbitrary order-independent options.
    // In practice may also be occasionally used for linking native libraries.
    // Passed after compiler-generated options to support manual overriding when necessary.
    add_user_defined_link_args(cmd, sess);

    // ------------ Object code and libraries, order-dependent ------------

    // Post-link CRT objects.
    add_post_link_objects(cmd, sess, link_output_kind, crt_objects_fallback);

    // ------------ Late order-dependent options ------------

    // Doesn't really make sense.
    // FIXME: In practice built-in target specs use this for arbitrary order-independent options,
    // introduce a target spec option for order-independent linker options, migrate built-in specs
    // to it and remove the option.
    add_post_link_args(cmd, sess, flavor);

    cmd.take_cmd()
}

fn add_order_independent_options(
    cmd: &mut dyn Linker,
    sess: &Session,
    link_output_kind: LinkOutputKind,
    crt_objects_fallback: bool,
    flavor: LinkerFlavor,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
    out_filename: &Path,
    tmpdir: &Path,
) {
    add_gcc_ld_path(cmd, sess, flavor);

    add_apple_sdk(cmd, sess, flavor);

    add_link_script(cmd, sess, tmpdir, crate_type);

    if sess.target.is_like_fuchsia && crate_type == CrateType::Executable {
        let prefix = if sess.opts.debugging_opts.sanitizer.contains(SanitizerSet::ADDRESS) {
            "asan/"
        } else {
            ""
        };
        cmd.arg(format!("--dynamic-linker={}ld.so.1", prefix));
    }

    if sess.target.eh_frame_header {
        cmd.add_eh_frame_header();
    }

    // Make the binary compatible with data execution prevention schemes.
    cmd.add_no_exec();

    if crt_objects_fallback {
        cmd.no_crt_objects();
    }

    if sess.target.is_like_emscripten {
        cmd.arg("-s");
        cmd.arg(if sess.panic_strategy() == PanicStrategy::Abort {
            "DISABLE_EXCEPTION_CATCHING=1"
        } else {
            "DISABLE_EXCEPTION_CATCHING=0"
        });
    }

    if flavor == LinkerFlavor::PtxLinker {
        // Provide the linker with fallback to internal `target-cpu`.
        cmd.arg("--fallback-arch");
        cmd.arg(&codegen_results.crate_info.target_cpu);
    } else if flavor == LinkerFlavor::BpfLinker {
        cmd.arg("--cpu");
        cmd.arg(&codegen_results.crate_info.target_cpu);
        cmd.arg("--cpu-features");
        cmd.arg(match &sess.opts.cg.target_feature {
            feat if !feat.is_empty() => feat,
            _ => &sess.target.options.features,
        });
    }

    cmd.linker_plugin_lto();

    add_library_search_dirs(cmd, sess, crt_objects_fallback);

    cmd.output_filename(out_filename);

    if crate_type == CrateType::Executable && sess.target.is_like_windows {
        if let Some(ref s) = codegen_results.crate_info.windows_subsystem {
            cmd.subsystem(s);
        }
    }

    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    if !sess.link_dead_code() {
        // If PGO is enabled sometimes gc_sections will remove the profile data section
        // as it appears to be unused. This can then cause the PGO profile file to lose
        // some functions. If we are generating a profile we shouldn't strip those metadata
        // sections to ensure we have all the data for PGO.
        let keep_metadata =
            crate_type == CrateType::Dylib || sess.opts.cg.profile_generate.enabled();
        cmd.gc_sections(keep_metadata);
    }

    cmd.set_output_kind(link_output_kind, out_filename);

    add_relro_args(cmd, sess);

    // Pass optimization flags down to the linker.
    cmd.optimize();

    // Pass debuginfo and strip flags down to the linker.
    cmd.debuginfo(sess.opts.debugging_opts.strip);

    // We want to prevent the compiler from accidentally leaking in any system libraries,
    // so by default we tell linkers not to link to any default libraries.
    if !sess.opts.cg.default_linker_libraries && sess.target.no_default_libraries {
        cmd.no_default_libraries();
    }

    if sess.opts.cg.profile_generate.enabled() || sess.instrument_coverage() {
        cmd.pgo_gen();
    }

    if sess.opts.cg.control_flow_guard != CFGuard::Disabled {
        cmd.control_flow_guard();
    }

    add_rpath_args(cmd, sess, codegen_results, out_filename);
}

/// # Native library linking
///
/// User-supplied library search paths (-L on the command line). These are the same paths used to
/// find Rust crates, so some of them may have been added already by the previous crate linking
/// code. This only allows them to be found at compile time so it is still entirely up to outside
/// forces to make sure that library can be found at runtime.
///
/// Also note that the native libraries linked here are only the ones located in the current crate.
/// Upstream crates with native library dependencies may have their native library pulled in above.
fn add_local_native_libraries(
    cmd: &mut dyn Linker,
    sess: &Session,
    codegen_results: &CodegenResults,
) {
    let filesearch = sess.target_filesearch(PathKind::All);
    for search_path in filesearch.search_paths() {
        match search_path.kind {
            PathKind::Framework => {
                cmd.framework_path(&search_path.dir);
            }
            _ => {
                cmd.include_path(&fix_windows_verbatim_for_gcc(&search_path.dir));
            }
        }
    }

    let relevant_libs =
        codegen_results.crate_info.used_libraries.iter().filter(|l| relevant_lib(sess, l));

    let search_path = OnceCell::new();
    let mut last = (NativeLibKind::Unspecified, None);
    for lib in relevant_libs {
        let name = match lib.name {
            Some(l) => l,
            None => continue,
        };

        // Skip if this library is the same as the last.
        last = if (lib.kind, lib.name) == last { continue } else { (lib.kind, lib.name) };

        let verbatim = lib.verbatim.unwrap_or(false);
        match lib.kind {
            NativeLibKind::Dylib { as_needed } => {
                cmd.link_dylib(name, verbatim, as_needed.unwrap_or(true))
            }
            NativeLibKind::Unspecified => cmd.link_dylib(name, verbatim, true),
            NativeLibKind::Framework { as_needed } => {
                cmd.link_framework(name, as_needed.unwrap_or(true))
            }
            NativeLibKind::Static { bundle: None | Some(true), .. }
            | NativeLibKind::Static { whole_archive: Some(true), .. } => {
                cmd.link_whole_staticlib(
                    name,
                    verbatim,
                    &search_path.get_or_init(|| archive_search_paths(sess)),
                );
            }
            NativeLibKind::Static { .. } => cmd.link_staticlib(name, verbatim),
            NativeLibKind::RawDylib => {
                // FIXME(#58713): Proper handling for raw dylibs.
                bug!("raw_dylib feature not yet implemented");
            }
        }
    }
}

/// # Linking Rust crates and their nobundle static libraries
///
/// Rust crates are not considered at all when creating an rlib output. All dependencies will be
/// linked when producing the final output (instead of the intermediate rlib version).
fn add_upstream_rust_crates<'a, B: ArchiveBuilder<'a>>(
    cmd: &mut dyn Linker,
    sess: &'a Session,
    codegen_results: &CodegenResults,
    crate_type: CrateType,
    tmpdir: &Path,
) {
    // All of the heavy lifting has previously been accomplished by the
    // dependency_format module of the compiler. This is just crawling the
    // output of that module, adding crates as necessary.
    //
    // Linking to a rlib involves just passing it to the linker (the linker
    // will slurp up the object files inside), and linking to a dynamic library
    // involves just passing the right -l flag.

    let (_, data) = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| *ty == crate_type)
        .expect("failed to find crate type in dependency format list");

    // Invoke get_used_crates to ensure that we get a topological sorting of
    // crates.
    let deps = &codegen_results.crate_info.used_crates;

    // There's a few internal crates in the standard library (aka libcore and
    // libstd) which actually have a circular dependence upon one another. This
    // currently arises through "weak lang items" where libcore requires things
    // like `rust_begin_unwind` but libstd ends up defining it. To get this
    // circular dependence to work correctly in all situations we'll need to be
    // sure to correctly apply the `--start-group` and `--end-group` options to
    // GNU linkers, otherwise if we don't use any other symbol from the standard
    // library it'll get discarded and the whole application won't link.
    //
    // In this loop we're calculating the `group_end`, after which crate to
    // pass `--end-group` and `group_start`, before which crate to pass
    // `--start-group`. We currently do this by passing `--end-group` after
    // the first crate (when iterating backwards) that requires a lang item
    // defined somewhere else. Once that's set then when we've defined all the
    // necessary lang items we'll pass `--start-group`.
    //
    // Note that this isn't amazing logic for now but it should do the trick
    // for the current implementation of the standard library.
    let mut group_end = None;
    let mut group_start = None;
    // Crates available for linking thus far.
    let mut available = FxHashSet::default();
    // Crates required to satisfy dependencies discovered so far.
    let mut required = FxHashSet::default();

    let info = &codegen_results.crate_info;
    for &cnum in deps.iter().rev() {
        if let Some(missing) = info.missing_lang_items.get(&cnum) {
            let missing_crates = missing.iter().map(|i| info.lang_item_to_crate.get(i).copied());
            required.extend(missing_crates);
        }

        required.insert(Some(cnum));
        available.insert(Some(cnum));

        if required.len() > available.len() && group_end.is_none() {
            group_end = Some(cnum);
        }
        if required.len() == available.len() && group_end.is_some() {
            group_start = Some(cnum);
            break;
        }
    }

    // If we didn't end up filling in all lang items from upstream crates then
    // we'll be filling it in with our crate. This probably means we're the
    // standard library itself, so skip this for now.
    if group_end.is_some() && group_start.is_none() {
        group_end = None;
    }

    let mut compiler_builtins = None;
    let search_path = OnceCell::new();

    for &cnum in deps.iter() {
        if group_start == Some(cnum) {
            cmd.group_start();
        }

        // We may not pass all crates through to the linker. Some crates may
        // appear statically in an existing dylib, meaning we'll pick up all the
        // symbols from the dylib.
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        match data[cnum.as_usize() - 1] {
            _ if codegen_results.crate_info.profiler_runtime == Some(cnum) => {
                add_static_crate::<B>(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
            }
            // compiler-builtins are always placed last to ensure that they're
            // linked correctly.
            _ if codegen_results.crate_info.compiler_builtins == Some(cnum) => {
                assert!(compiler_builtins.is_none());
                compiler_builtins = Some(cnum);
            }
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                add_static_crate::<B>(cmd, sess, codegen_results, tmpdir, crate_type, cnum);

                // Link static native libs with "-bundle" modifier only if the crate they originate from
                // is being linked statically to the current crate.  If it's linked dynamically
                // or is an rlib already included via some other dylib crate, the symbols from
                // native libs will have already been included in that dylib.
                //
                // If -Zlink-native-libraries=false is set, then the assumption is that an
                // external build system already has the native dependencies defined, and it
                // will provide them to the linker itself.
                if sess.opts.debugging_opts.link_native_libraries {
                    let mut last = None;
                    for lib in &codegen_results.crate_info.native_libraries[&cnum] {
                        if !relevant_lib(sess, lib) {
                            // Skip libraries if they are disabled by `#[link(cfg=...)]`
                            continue;
                        }

                        // Skip if this library is the same as the last.
                        if last == lib.name {
                            continue;
                        }

                        if let Some(static_lib_name) = lib.name {
                            if let NativeLibKind::Static { bundle: Some(false), whole_archive } =
                                lib.kind
                            {
                                let verbatim = lib.verbatim.unwrap_or(false);
                                if whole_archive == Some(true) {
                                    cmd.link_whole_staticlib(
                                        static_lib_name,
                                        verbatim,
                                        search_path.get_or_init(|| archive_search_paths(sess)),
                                    );
                                } else {
                                    cmd.link_staticlib(static_lib_name, verbatim);
                                }

                                last = lib.name;
                            }
                        }
                    }
                }
            }
            Linkage::Dynamic => add_dynamic_crate(cmd, sess, &src.dylib.as_ref().unwrap().0),
        }

        if group_end == Some(cnum) {
            cmd.group_end();
        }
    }

    // compiler-builtins are always placed last to ensure that they're
    // linked correctly.
    // We must always link the `compiler_builtins` crate statically. Even if it
    // was already "included" in a dylib (e.g., `libstd` when `-C prefer-dynamic`
    // is used)
    if let Some(cnum) = compiler_builtins {
        add_static_crate::<B>(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
    }

    // Converts a library file-stem into a cc -l argument
    fn unlib<'a>(target: &Target, stem: &'a str) -> &'a str {
        if stem.starts_with("lib") && !target.is_like_windows { &stem[3..] } else { stem }
    }

    // Adds the static "rlib" versions of all crates to the command line.
    // There's a bit of magic which happens here specifically related to LTO,
    // namely that we remove upstream object files.
    //
    // When performing LTO, almost(*) all of the bytecode from the upstream
    // libraries has already been included in our object file output. As a
    // result we need to remove the object files in the upstream libraries so
    // the linker doesn't try to include them twice (or whine about duplicate
    // symbols). We must continue to include the rest of the rlib, however, as
    // it may contain static native libraries which must be linked in.
    //
    // (*) Crates marked with `#![no_builtins]` don't participate in LTO and
    // their bytecode wasn't included. The object files in those libraries must
    // still be passed to the linker.
    //
    // Note, however, that if we're not doing LTO we can just pass the rlib
    // blindly to the linker (fast) because it's fine if it's not actually
    // included as we're at the end of the dependency chain.
    fn add_static_crate<'a, B: ArchiveBuilder<'a>>(
        cmd: &mut dyn Linker,
        sess: &'a Session,
        codegen_results: &CodegenResults,
        tmpdir: &Path,
        crate_type: CrateType,
        cnum: CrateNum,
    ) {
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        let cratepath = &src.rlib.as_ref().unwrap().0;

        let mut link_upstream = |path: &Path| {
            // If we're creating a dylib, then we need to include the
            // whole of each object in our archive into that artifact. This is
            // because a `dylib` can be reused as an intermediate artifact.
            //
            // Note, though, that we don't want to include the whole of a
            // compiler-builtins crate (e.g., compiler-rt) because it'll get
            // repeatedly linked anyway.
            let path = fix_windows_verbatim_for_gcc(path);
            if crate_type == CrateType::Dylib
                && codegen_results.crate_info.compiler_builtins != Some(cnum)
            {
                cmd.link_whole_rlib(&path);
            } else {
                cmd.link_rlib(&path);
            }
        };

        // See the comment above in `link_staticlib` and `link_rlib` for why if
        // there's a static library that's not relevant we skip all object
        // files.
        let native_libs = &codegen_results.crate_info.native_libraries[&cnum];
        let skip_native = native_libs.iter().any(|lib| {
            matches!(lib.kind, NativeLibKind::Static { bundle: None | Some(true), .. })
                && !relevant_lib(sess, lib)
        });

        if (!are_upstream_rust_objects_already_included(sess)
            || ignored_for_lto(sess, &codegen_results.crate_info, cnum))
            && !skip_native
        {
            link_upstream(cratepath);
            return;
        }

        let dst = tmpdir.join(cratepath.file_name().unwrap());
        let name = cratepath.file_name().unwrap().to_str().unwrap();
        let name = &name[3..name.len() - 5]; // chop off lib/.rlib

        sess.prof.generic_activity_with_arg("link_altering_rlib", name).run(|| {
            let mut archive = <B as ArchiveBuilder>::new(sess, &dst, Some(cratepath));
            archive.update_symbols();

            let mut any_objects = false;
            for f in archive.src_files() {
                if f == METADATA_FILENAME {
                    archive.remove_file(&f);
                    continue;
                }

                let canonical = f.replace("-", "_");
                let canonical_name = name.replace("-", "_");

                let is_rust_object =
                    canonical.starts_with(&canonical_name) && looks_like_rust_object_file(&f);

                // If we've been requested to skip all native object files
                // (those not generated by the rust compiler) then we can skip
                // this file. See above for why we may want to do this.
                let skip_because_cfg_say_so = skip_native && !is_rust_object;

                // If we're performing LTO and this is a rust-generated object
                // file, then we don't need the object file as it's part of the
                // LTO module. Note that `#![no_builtins]` is excluded from LTO,
                // though, so we let that object file slide.
                let skip_because_lto = are_upstream_rust_objects_already_included(sess)
                    && is_rust_object
                    && (sess.target.no_builtins
                        || !codegen_results.crate_info.is_no_builtins.contains(&cnum));

                if skip_because_cfg_say_so || skip_because_lto {
                    archive.remove_file(&f);
                } else {
                    any_objects = true;
                }
            }

            if !any_objects {
                return;
            }
            archive.build();
            link_upstream(&dst);
        });
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crate(cmd: &mut dyn Linker, sess: &Session, cratepath: &Path) {
        // Just need to tell the linker about where the library lives and
        // what its name is
        let parent = cratepath.parent();
        if let Some(dir) = parent {
            cmd.include_path(&fix_windows_verbatim_for_gcc(dir));
        }
        let filestem = cratepath.file_stem().unwrap().to_str().unwrap();
        cmd.link_rust_dylib(
            Symbol::intern(&unlib(&sess.target, filestem)),
            parent.unwrap_or_else(|| Path::new("")),
        );
    }
}

/// Link in all of our upstream crates' native dependencies. Remember that all of these upstream
/// native dependencies are all non-static dependencies. We've got two cases then:
///
/// 1. The upstream crate is an rlib. In this case we *must* link in the native dependency because
/// the rlib is just an archive.
///
/// 2. The upstream crate is a dylib. In order to use the dylib, we have to have the dependency
/// present on the system somewhere. Thus, we don't gain a whole lot from not linking in the
/// dynamic dependency to this crate as well.
///
/// The use case for this is a little subtle. In theory the native dependencies of a crate are
/// purely an implementation detail of the crate itself, but the problem arises with generic and
/// inlined functions. If a generic function calls a native function, then the generic function
/// must be instantiated in the target crate, meaning that the native symbol must also be resolved
/// in the target crate.
fn add_upstream_native_libraries(
    cmd: &mut dyn Linker,
    sess: &Session,
    codegen_results: &CodegenResults,
) {
    let mut last = (NativeLibKind::Unspecified, None);
    for &cnum in &codegen_results.crate_info.used_crates {
        for lib in codegen_results.crate_info.native_libraries[&cnum].iter() {
            let name = match lib.name {
                Some(l) => l,
                None => continue,
            };
            if !relevant_lib(sess, &lib) {
                continue;
            }

            // Skip if this library is the same as the last.
            last = if (lib.kind, lib.name) == last { continue } else { (lib.kind, lib.name) };

            let verbatim = lib.verbatim.unwrap_or(false);
            match lib.kind {
                NativeLibKind::Dylib { as_needed } => {
                    cmd.link_dylib(name, verbatim, as_needed.unwrap_or(true))
                }
                NativeLibKind::Unspecified => cmd.link_dylib(name, verbatim, true),
                NativeLibKind::Framework { as_needed } => {
                    cmd.link_framework(name, as_needed.unwrap_or(true))
                }
                // ignore static native libraries here as we've
                // already included them in add_local_native_libraries and
                // add_upstream_rust_crates
                NativeLibKind::Static { .. } => {}
                NativeLibKind::RawDylib => {}
            }
        }
    }
}

fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => rustc_attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

fn are_upstream_rust_objects_already_included(sess: &Session) -> bool {
    match sess.lto() {
        config::Lto::Fat => true,
        config::Lto::Thin => {
            // If we defer LTO to the linker, we haven't run LTO ourselves, so
            // any upstream object files have not been copied yet.
            !sess.opts.cg.linker_plugin_lto.enabled()
        }
        config::Lto::No | config::Lto::ThinLocal => false,
    }
}

fn add_apple_sdk(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    let arch = &sess.target.arch;
    let os = &sess.target.os;
    let llvm_target = &sess.target.llvm_target;
    if sess.target.vendor != "apple"
        || !matches!(os.as_str(), "ios" | "tvos")
        || flavor != LinkerFlavor::Gcc
    {
        return;
    }
    let sdk_name = match (arch.as_str(), os.as_str()) {
        ("aarch64", "tvos") => "appletvos",
        ("x86_64", "tvos") => "appletvsimulator",
        ("arm", "ios") => "iphoneos",
        ("aarch64", "ios") if llvm_target.contains("macabi") => "macosx",
        ("aarch64", "ios") if llvm_target.contains("sim") => "iphonesimulator",
        ("aarch64", "ios") => "iphoneos",
        ("x86", "ios") => "iphonesimulator",
        ("x86_64", "ios") if llvm_target.contains("macabi") => "macosx",
        ("x86_64", "ios") => "iphonesimulator",
        _ => {
            sess.err(&format!("unsupported arch `{}` for os `{}`", arch, os));
            return;
        }
    };
    let sdk_root = match get_apple_sdk_root(sdk_name) {
        Ok(s) => s,
        Err(e) => {
            sess.err(&e);
            return;
        }
    };
    if llvm_target.contains("macabi") {
        cmd.args(&["-target", llvm_target])
    } else {
        let arch_name = llvm_target.split('-').next().expect("LLVM target must have a hyphen");
        cmd.args(&["-arch", arch_name])
    }
    cmd.args(&["-isysroot", &sdk_root, "-Wl,-syslibroot", &sdk_root]);
}

fn get_apple_sdk_root(sdk_name: &str) -> Result<String, String> {
    // Following what clang does
    // (https://github.com/llvm/llvm-project/blob/
    // 296a80102a9b72c3eda80558fb78a3ed8849b341/clang/lib/Driver/ToolChains/Darwin.cpp#L1661-L1678)
    // to allow the SDK path to be set. (For clang, xcrun sets
    // SDKROOT; for rustc, the user or build system can set it, or we
    // can fall back to checking for xcrun on PATH.)
    if let Ok(sdkroot) = env::var("SDKROOT") {
        let p = Path::new(&sdkroot);
        match sdk_name {
            // Ignore `SDKROOT` if it's clearly set for the wrong platform.
            "appletvos"
                if sdkroot.contains("TVSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "appletvsimulator"
                if sdkroot.contains("TVOS.platform") || sdkroot.contains("MacOSX.platform") => {}
            "iphoneos"
                if sdkroot.contains("iPhoneSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "iphonesimulator"
                if sdkroot.contains("iPhoneOS.platform") || sdkroot.contains("MacOSX.platform") => {
            }
            "macosx10.15"
                if sdkroot.contains("iPhoneOS.platform")
                    || sdkroot.contains("iPhoneSimulator.platform") => {}
            // Ignore `SDKROOT` if it's not a valid path.
            _ if !p.is_absolute() || p == Path::new("/") || !p.exists() => {}
            _ => return Ok(sdkroot),
        }
    }
    let res =
        Command::new("xcrun").arg("--show-sdk-path").arg("-sdk").arg(sdk_name).output().and_then(
            |output| {
                if output.status.success() {
                    Ok(String::from_utf8(output.stdout).unwrap())
                } else {
                    let error = String::from_utf8(output.stderr);
                    let error = format!("process exit with error: {}", error.unwrap());
                    Err(io::Error::new(io::ErrorKind::Other, &error[..]))
                }
            },
        );

    match res {
        Ok(output) => Ok(output.trim().to_string()),
        Err(e) => Err(format!("failed to get {} SDK path: {}", sdk_name, e)),
    }
}

fn add_gcc_ld_path(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    if let Some(ld_impl) = sess.opts.debugging_opts.gcc_ld {
        if let LinkerFlavor::Gcc = flavor {
            match ld_impl {
                LdImpl::Lld => {
                    if sess.target.lld_flavor == LldFlavor::Ld64 {
                        let tools_path = sess.get_tools_search_paths(false);
                        let ld64_exe = tools_path
                            .into_iter()
                            .map(|p| p.join("gcc-ld"))
                            .map(|p| {
                                p.join(if sess.host.is_like_windows { "ld64.exe" } else { "ld64" })
                            })
                            .find(|p| p.exists())
                            .unwrap_or_else(|| sess.fatal("rust-lld (as ld64) not found"));
                        cmd.cmd().arg({
                            let mut arg = OsString::from("-fuse-ld=");
                            arg.push(ld64_exe);
                            arg
                        });
                    } else {
                        let tools_path = sess.get_tools_search_paths(false);
                        let lld_path = tools_path
                            .into_iter()
                            .map(|p| p.join("gcc-ld"))
                            .find(|p| {
                                p.join(if sess.host.is_like_windows { "ld.exe" } else { "ld" })
                                    .exists()
                            })
                            .unwrap_or_else(|| sess.fatal("rust-lld (as ld) not found"));
                        cmd.cmd().arg({
                            let mut arg = OsString::from("-B");
                            arg.push(lld_path);
                            arg
                        });
                    }
                }
            }
        } else {
            sess.fatal("option `-Z gcc-ld` is used even though linker flavor is not gcc");
        }
    }
}
