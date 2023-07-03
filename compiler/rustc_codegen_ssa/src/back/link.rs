use rustc_arena::TypedArena;
use rustc_ast::CRATE_NODE_ID;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_errors::{ErrorGuaranteed, Handler};
use rustc_fs_util::{fix_windows_verbatim_for_gcc, try_canonicalize};
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_metadata::find_native_static_library;
use rustc_metadata::fs::{copy_to_stdout, emit_wrapper_file, METADATA_FILENAME};
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_session::config::{self, CFGuard, CrateType, DebugInfo, Strip};
use rustc_session::config::{OutputFilenames, OutputType, PrintRequest, SplitDwarfKind};
use rustc_session::cstore::DllImport;
use rustc_session::output::{check_file_is_writeable, invalid_output_for_target, out_filename};
use rustc_session::search_paths::PathKind;
use rustc_session::utils::NativeLibKind;
/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.
use rustc_session::{filesearch, Session};
use rustc_span::symbol::Symbol;
use rustc_target::spec::crt_objects::{CrtObjects, LinkSelfContainedDefault};
use rustc_target::spec::{Cc, LinkOutputKind, LinkerFlavor, Lld, PanicStrategy};
use rustc_target::spec::{RelocModel, RelroLevel, SanitizerSet, SplitDebuginfo};

use super::archive::{ArchiveBuilder, ArchiveBuilderBuilder};
use super::command::Command;
use super::linker::{self, Linker};
use super::metadata::{create_wrapper_file, MetadataPosition};
use super::rpath::{self, RPathConfig};
use crate::{
    errors, looks_like_rust_object_file, CodegenResults, CompiledModule, CrateInfo, NativeLib,
};

use cc::windows_registry;
use regex::Regex;
use tempfile::Builder as TempFileBuilder;

use itertools::Itertools;
use std::cell::OnceCell;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{read, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};
use std::{env, fmt, fs, io, mem, str};

pub fn ensure_removed(diag_handler: &Handler, path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            diag_handler.err(format!("failed to remove {}: {}", path.display(), e));
        }
    }
}

/// Performs the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary<'a>(
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    outputs: &OutputFilenames,
) -> Result<(), ErrorGuaranteed> {
    let _timer = sess.timer("link_binary");
    let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
    let mut tempfiles_for_stdout_output: Vec<PathBuf> = Vec::new();
    for &crate_type in sess.crate_types().iter() {
        // Ignore executable crates if we have -Z no-codegen, as they will error.
        if (sess.opts.unstable_opts.no_codegen || !sess.opts.output_types.should_codegen())
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
                .unwrap_or_else(|error| sess.emit_fatal(errors::CreateTempDir { error }));
            let path = MaybeTempDir::new(tmpdir, sess.opts.cg.save_temps);
            let output = out_filename(
                sess,
                crate_type,
                outputs,
                codegen_results.crate_info.local_crate_name,
            );
            let crate_name = format!("{}", codegen_results.crate_info.local_crate_name);
            let out_filename =
                output.file_for_writing(outputs, OutputType::Exe, Some(crate_name.as_str()));
            match crate_type {
                CrateType::Rlib => {
                    let _timer = sess.timer("link_rlib");
                    info!("preparing rlib to {:?}", out_filename);
                    link_rlib(
                        sess,
                        archive_builder_builder,
                        codegen_results,
                        RlibFlavor::Normal,
                        &path,
                    )?
                    .build(&out_filename);
                }
                CrateType::Staticlib => {
                    link_staticlib(
                        sess,
                        archive_builder_builder,
                        codegen_results,
                        &out_filename,
                        &path,
                    )?;
                }
                _ => {
                    link_natively(
                        sess,
                        archive_builder_builder,
                        crate_type,
                        &out_filename,
                        codegen_results,
                        path.as_ref(),
                    )?;
                }
            }
            if sess.opts.json_artifact_notifications {
                sess.parse_sess.span_diagnostic.emit_artifact_notification(&out_filename, "link");
            }

            if sess.prof.enabled() {
                if let Some(artifact_name) = out_filename.file_name() {
                    // Record size for self-profiling
                    let file_size = std::fs::metadata(&out_filename).map(|m| m.len()).unwrap_or(0);

                    sess.prof.artifact_size(
                        "linked_artifact",
                        artifact_name.to_string_lossy(),
                        file_size,
                    );
                }
            }

            if output.is_stdout() {
                if output.is_tty() {
                    sess.emit_err(errors::BinaryOutputToTty {
                        shorthand: OutputType::Exe.shorthand(),
                    });
                } else if let Err(e) = copy_to_stdout(&out_filename) {
                    sess.emit_err(errors::CopyPath::new(&out_filename, output.as_path(), e));
                }
                tempfiles_for_stdout_output.push(out_filename);
            }
        }
    }

    // Remove the temporary object file and metadata if we aren't saving temps.
    sess.time("link_binary_remove_temps", || {
        // If the user requests that temporaries are saved, don't delete any.
        if sess.opts.cg.save_temps {
            return;
        }

        let maybe_remove_temps_from_module =
            |preserve_objects: bool, preserve_dwarf_objects: bool, module: &CompiledModule| {
                if !preserve_objects {
                    if let Some(ref obj) = module.object {
                        ensure_removed(sess.diagnostic(), obj);
                    }
                }

                if !preserve_dwarf_objects {
                    if let Some(ref dwo_obj) = module.dwarf_object {
                        ensure_removed(sess.diagnostic(), dwo_obj);
                    }
                }
            };

        let remove_temps_from_module =
            |module: &CompiledModule| maybe_remove_temps_from_module(false, false, module);

        // Otherwise, always remove the metadata and allocator module temporaries.
        if let Some(ref metadata_module) = codegen_results.metadata_module {
            remove_temps_from_module(metadata_module);
        }

        if let Some(ref allocator_module) = codegen_results.allocator_module {
            remove_temps_from_module(allocator_module);
        }

        // Remove the temporary files if output goes to stdout
        for temp in tempfiles_for_stdout_output {
            ensure_removed(sess.diagnostic(), &temp);
        }

        // If no requested outputs require linking, then the object temporaries should
        // be kept.
        if !sess.opts.output_types.should_link() {
            return;
        }

        // Potentially keep objects for their debuginfo.
        let (preserve_objects, preserve_dwarf_objects) = preserve_objects_for_their_debuginfo(sess);
        debug!(?preserve_objects, ?preserve_dwarf_objects);

        for module in &codegen_results.modules {
            maybe_remove_temps_from_module(preserve_objects, preserve_dwarf_objects, module);
        }
    });

    Ok(())
}

// Crate type is not passed when calculating the dylibs to include for LTO. In that case all
// crate types must use the same dependency formats.
pub fn each_linked_rlib(
    info: &CrateInfo,
    crate_type: Option<CrateType>,
    f: &mut dyn FnMut(CrateNum, &Path),
) -> Result<(), errors::LinkRlibError> {
    let crates = info.used_crates.iter();

    let fmts = if crate_type.is_none() {
        for combination in info.dependency_formats.iter().combinations(2) {
            let (ty1, list1) = &combination[0];
            let (ty2, list2) = &combination[1];
            if list1 != list2 {
                return Err(errors::LinkRlibError::IncompatibleDependencyFormats {
                    ty1: format!("{ty1:?}"),
                    ty2: format!("{ty2:?}"),
                    list1: format!("{list1:?}"),
                    list2: format!("{list2:?}"),
                });
            }
        }
        if info.dependency_formats.is_empty() {
            return Err(errors::LinkRlibError::MissingFormat);
        }
        &info.dependency_formats[0].1
    } else {
        let fmts = info
            .dependency_formats
            .iter()
            .find_map(|&(ty, ref list)| if Some(ty) == crate_type { Some(list) } else { None });

        let Some(fmts) = fmts else {
            return Err(errors::LinkRlibError::MissingFormat);
        };

        fmts
    };

    for &cnum in crates {
        match fmts.get(cnum.as_usize() - 1) {
            Some(&Linkage::NotLinked | &Linkage::Dynamic | &Linkage::IncludedFromDylib) => continue,
            Some(_) => {}
            None => return Err(errors::LinkRlibError::MissingFormat),
        }
        let crate_name = info.crate_name[&cnum];
        let used_crate_source = &info.used_crate_source[&cnum];
        if let Some((path, _)) = &used_crate_source.rlib {
            f(cnum, &path);
        } else {
            if used_crate_source.rmeta.is_some() {
                return Err(errors::LinkRlibError::OnlyRmetaFound { crate_name });
            } else {
                return Err(errors::LinkRlibError::NotFound { crate_name });
            }
        }
    }
    Ok(())
}

/// Create an 'rlib'.
///
/// An rlib in its current incarnation is essentially a renamed .a file (with "dummy" object files).
/// The rlib primarily contains the object file of the crate, but it also some of the object files
/// from native libraries.
fn link_rlib<'a>(
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    flavor: RlibFlavor,
    tmpdir: &MaybeTempDir,
) -> Result<Box<dyn ArchiveBuilder<'a> + 'a>, ErrorGuaranteed> {
    let lib_search_paths = archive_search_paths(sess);

    let mut ab = archive_builder_builder.new_archive_builder(sess);

    let trailing_metadata = match flavor {
        RlibFlavor::Normal => {
            let (metadata, metadata_position) =
                create_wrapper_file(sess, b".rmeta".to_vec(), codegen_results.metadata.raw_data());
            let metadata = emit_wrapper_file(sess, &metadata, tmpdir, METADATA_FILENAME);
            match metadata_position {
                MetadataPosition::First => {
                    // Most of the time metadata in rlib files is wrapped in a "dummy" object
                    // file for the target platform so the rlib can be processed entirely by
                    // normal linkers for the platform. Sometimes this is not possible however.
                    // If it is possible however, placing the metadata object first improves
                    // performance of getting metadata from rlibs.
                    ab.add_file(&metadata);
                    None
                }
                MetadataPosition::Last => Some(metadata),
            }
        }

        RlibFlavor::StaticlibBase => None,
    };

    for m in &codegen_results.modules {
        if let Some(obj) = m.object.as_ref() {
            ab.add_file(obj);
        }

        if let Some(dwarf_obj) = m.dwarf_object.as_ref() {
            ab.add_file(dwarf_obj);
        }
    }

    match flavor {
        RlibFlavor::Normal => {}
        RlibFlavor::StaticlibBase => {
            let obj = codegen_results.allocator_module.as_ref().and_then(|m| m.object.as_ref());
            if let Some(obj) = obj {
                ab.add_file(obj);
            }
        }
    }

    // Used if packed_bundled_libs flag enabled.
    let mut packed_bundled_libs = Vec::new();

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
        let NativeLibKind::Static { bundle: None | Some(true), whole_archive } = lib.kind else {
            continue;
        };
        if whole_archive == Some(true)
            && flavor == RlibFlavor::Normal
            && !codegen_results.crate_info.feature_packed_bundled_libs
        {
            sess.emit_err(errors::IncompatibleLinkingModifiers);
        }
        if flavor == RlibFlavor::Normal && let Some(filename) = lib.filename {
            let path = find_native_static_library(filename.as_str(), true, &lib_search_paths, sess);
            let src = read(path).map_err(|e| sess.emit_fatal(errors::ReadFileError {message: e }))?;
            let (data, _) = create_wrapper_file(sess, b".bundled_lib".to_vec(), &src);
            let wrapper_file = emit_wrapper_file(sess, &data, tmpdir, filename.as_str());
            packed_bundled_libs.push(wrapper_file);
        } else {
            let path =
                find_native_static_library(lib.name.as_str(), lib.verbatim, &lib_search_paths, sess);
            ab.add_archive(&path, Box::new(|_| false)).unwrap_or_else(|error| {
                sess.emit_fatal(errors::AddNativeLibrary { library_path: path, error })});
        }
    }

    for (raw_dylib_name, raw_dylib_imports) in
        collate_raw_dylibs(sess, codegen_results.crate_info.used_libraries.iter())?
    {
        let output_path = archive_builder_builder.create_dll_import_lib(
            sess,
            &raw_dylib_name,
            &raw_dylib_imports,
            tmpdir.as_ref(),
            true,
        );

        ab.add_archive(&output_path, Box::new(|_| false)).unwrap_or_else(|error| {
            sess.emit_fatal(errors::AddNativeLibrary { library_path: output_path, error });
        });
    }

    if let Some(trailing_metadata) = trailing_metadata {
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
        // * Most of the time metadata in rlib files is wrapped in a "dummy" object
        //   file for the target platform so the rlib can be processed entirely by
        //   normal linkers for the platform. Sometimes this is not possible however.
        //
        // Basically, all this means is that this code should not move above the
        // code above.
        ab.add_file(&trailing_metadata);
    }

    // Add all bundled static native library dependencies.
    // Archives added to the end of .rlib archive, see comment above for the reason.
    for lib in packed_bundled_libs {
        ab.add_file(&lib)
    }

    return Ok(ab);
}

/// Extract all symbols defined in raw-dylib libraries, collated by library name.
///
/// If we have multiple extern blocks that specify symbols defined in the same raw-dylib library,
/// then the CodegenResults value contains one NativeLib instance for each block. However, the
/// linker appears to expect only a single import library for each library used, so we need to
/// collate the symbols together by library name before generating the import libraries.
fn collate_raw_dylibs<'a, 'b>(
    sess: &'a Session,
    used_libraries: impl IntoIterator<Item = &'b NativeLib>,
) -> Result<Vec<(String, Vec<DllImport>)>, ErrorGuaranteed> {
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
                        sess.emit_err(errors::MultipleExternalFuncDecl {
                            span: import.span,
                            function: import.name,
                            library_name: &name,
                        });
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
fn link_staticlib<'a>(
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    out_filename: &Path,
    tempdir: &MaybeTempDir,
) -> Result<(), ErrorGuaranteed> {
    info!("preparing staticlib to {:?}", out_filename);
    let mut ab = link_rlib(
        sess,
        archive_builder_builder,
        codegen_results,
        RlibFlavor::StaticlibBase,
        tempdir,
    )?;
    let mut all_native_libs = vec![];

    let res = each_linked_rlib(
        &codegen_results.crate_info,
        Some(CrateType::Staticlib),
        &mut |cnum, path| {
            let lto = are_upstream_rust_objects_already_included(sess)
                && !ignored_for_lto(sess, &codegen_results.crate_info, cnum);

            let native_libs = codegen_results.crate_info.native_libraries[&cnum].iter();
            let relevant = native_libs.clone().filter(|lib| relevant_lib(sess, &lib));
            let relevant_libs: FxHashSet<_> = relevant.filter_map(|lib| lib.filename).collect();

            let bundled_libs: FxHashSet<_> = native_libs.filter_map(|lib| lib.filename).collect();
            ab.add_archive(
                path,
                Box::new(move |fname: &str| {
                    // Ignore metadata files, no matter the name.
                    if fname == METADATA_FILENAME {
                        return true;
                    }

                    // Don't include Rust objects if LTO is enabled
                    if lto && looks_like_rust_object_file(fname) {
                        return true;
                    }

                    // Skip objects for bundled libs.
                    if bundled_libs.contains(&Symbol::intern(fname)) {
                        return true;
                    }

                    false
                }),
            )
            .unwrap();

            archive_builder_builder
                .extract_bundled_libs(path, tempdir.as_ref(), &relevant_libs)
                .unwrap_or_else(|e| sess.emit_fatal(e));
            for filename in relevant_libs {
                let joined = tempdir.as_ref().join(filename.as_str());
                let path = joined.as_path();
                ab.add_archive(path, Box::new(|_| false)).unwrap();
            }

            all_native_libs
                .extend(codegen_results.crate_info.native_libraries[&cnum].iter().cloned());
        },
    );
    if let Err(e) = res {
        sess.emit_fatal(e);
    }

    ab.build(out_filename);

    let crates = codegen_results.crate_info.used_crates.iter();

    let fmts = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find_map(|&(ty, ref list)| if ty == CrateType::Staticlib { Some(list) } else { None })
        .expect("no dependency formats for staticlib");

    let mut all_rust_dylibs = vec![];
    for &cnum in crates {
        match fmts.get(cnum.as_usize() - 1) {
            Some(&Linkage::Dynamic) => {}
            _ => continue,
        }
        let crate_name = codegen_results.crate_info.crate_name[&cnum];
        let used_crate_source = &codegen_results.crate_info.used_crate_source[&cnum];
        if let Some((path, _)) = &used_crate_source.dylib {
            all_rust_dylibs.push(&**path);
        } else {
            if used_crate_source.rmeta.is_some() {
                sess.emit_fatal(errors::LinkRlibError::OnlyRmetaFound { crate_name });
            } else {
                sess.emit_fatal(errors::LinkRlibError::NotFound { crate_name });
            }
        }
    }

    all_native_libs.extend_from_slice(&codegen_results.crate_info.used_libraries);

    if sess.opts.prints.contains(&PrintRequest::NativeStaticLibs) {
        print_native_static_libs(sess, &all_native_libs, &all_rust_dylibs);
    }

    Ok(())
}

/// Use `thorin` (rust implementation of a dwarf packaging utility) to link DWARF objects into a
/// DWARF package.
fn link_dwarf_object<'a>(
    sess: &'a Session,
    cg_results: &CodegenResults,
    executable_out_filename: &Path,
) {
    let mut dwp_out_filename = executable_out_filename.to_path_buf().into_os_string();
    dwp_out_filename.push(".dwp");
    debug!(?dwp_out_filename, ?executable_out_filename);

    #[derive(Default)]
    struct ThorinSession<Relocations> {
        arena_data: TypedArena<Vec<u8>>,
        arena_mmap: TypedArena<Mmap>,
        arena_relocations: TypedArena<Relocations>,
    }

    impl<Relocations> ThorinSession<Relocations> {
        fn alloc_mmap(&self, data: Mmap) -> &Mmap {
            &*self.arena_mmap.alloc(data)
        }
    }

    impl<Relocations> thorin::Session<Relocations> for ThorinSession<Relocations> {
        fn alloc_data(&self, data: Vec<u8>) -> &[u8] {
            &*self.arena_data.alloc(data)
        }

        fn alloc_relocation(&self, data: Relocations) -> &Relocations {
            &*self.arena_relocations.alloc(data)
        }

        fn read_input(&self, path: &Path) -> std::io::Result<&[u8]> {
            let file = File::open(&path)?;
            let mmap = (unsafe { Mmap::map(file) })?;
            Ok(self.alloc_mmap(mmap))
        }
    }

    match sess.time("run_thorin", || -> Result<(), thorin::Error> {
        let thorin_sess = ThorinSession::default();
        let mut package = thorin::DwarfPackage::new(&thorin_sess);

        // Input objs contain .o/.dwo files from the current crate.
        match sess.opts.unstable_opts.split_dwarf_kind {
            SplitDwarfKind::Single => {
                for input_obj in cg_results.modules.iter().filter_map(|m| m.object.as_ref()) {
                    package.add_input_object(input_obj)?;
                }
            }
            SplitDwarfKind::Split => {
                for input_obj in cg_results.modules.iter().filter_map(|m| m.dwarf_object.as_ref()) {
                    package.add_input_object(input_obj)?;
                }
            }
        }

        // Input rlibs contain .o/.dwo files from dependencies.
        let input_rlibs = cg_results
            .crate_info
            .used_crate_source
            .values()
            .filter_map(|csource| csource.rlib.as_ref())
            .map(|(path, _)| path);
        for input_rlib in input_rlibs {
            debug!(?input_rlib);
            package.add_input_object(input_rlib)?;
        }

        // Failing to read the referenced objects is expected for dependencies where the path in the
        // executable will have been cleaned by Cargo, but the referenced objects will be contained
        // within rlibs provided as inputs.
        //
        // If paths have been remapped, then .o/.dwo files from the current crate also won't be
        // found, but are provided explicitly above.
        //
        // Adding an executable is primarily done to make `thorin` check that all the referenced
        // dwarf objects are found in the end.
        package.add_executable(
            &executable_out_filename,
            thorin::MissingReferencedObjectBehaviour::Skip,
        )?;

        let output_stream = BufWriter::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(dwp_out_filename)?,
        );
        let mut output_stream = object::write::StreamingBuffer::new(output_stream);
        package.finish()?.emit(&mut output_stream)?;
        output_stream.result()?;
        output_stream.into_inner().flush()?;

        Ok(())
    }) {
        Ok(()) => {}
        Err(e) => {
            sess.emit_err(errors::ThorinErrorWrapper(e));
            sess.abort_if_errors();
        }
    }
}

/// Create a dynamic library or executable.
///
/// This will invoke the system linker/cc to create the resulting file. This links to all upstream
/// files as well.
fn link_natively<'a>(
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    crate_type: CrateType,
    out_filename: &Path,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
) -> Result<(), ErrorGuaranteed> {
    info!("preparing {:?} to {:?}", crate_type, out_filename);
    let (linker_path, flavor) = linker_and_flavor(sess);
    let mut cmd = linker_with_args(
        &linker_path,
        flavor,
        sess,
        archive_builder_builder,
        crate_type,
        tmpdir,
        out_filename,
        codegen_results,
    )?;

    linker::disable_localization(&mut cmd);

    for (k, v) in sess.target.link_env.as_ref() {
        cmd.env(k.as_ref(), v.as_ref());
    }
    for k in sess.target.link_env_remove.as_ref() {
        cmd.env_remove(k.as_ref());
    }

    if sess.opts.prints.contains(&PrintRequest::LinkArgs) {
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
        let Ok(ref output) = prog else {
            break;
        };
        if output.status.success() {
            break;
        }
        let mut out = output.stderr.clone();
        out.extend(&output.stdout);
        let out = String::from_utf8_lossy(&out);

        // Check to see if the link failed with an error message that indicates it
        // doesn't recognize the -no-pie option. If so, re-perform the link step
        // without it. This is safe because if the linker doesn't support -no-pie
        // then it should not default to linking executables as pie. Different
        // versions of gcc seem to use different quotes in the error message so
        // don't check for them.
        if matches!(flavor, LinkerFlavor::Gnu(Cc::Yes, _))
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
        if matches!(flavor, LinkerFlavor::Gnu(Cc::Yes, _))
            && unknown_arg_regex.is_match(&out)
            && (out.contains("-static-pie") || out.contains("--no-dynamic-linker"))
            && cmd.get_args().iter().any(|e| e.to_string_lossy() == "-static-pie")
        {
            info!("linker output: {:?}", out);
            warn!(
                "Linker does not support -static-pie command line option. Retrying with -static instead."
            );
            // Mirror `add_(pre,post)_link_objects` to replace CRT objects.
            let self_contained = self_contained(sess, crate_type);
            let opts = &sess.target;
            let pre_objects = if self_contained {
                &opts.pre_link_objects_self_contained
            } else {
                &opts.pre_link_objects
            };
            let post_objects = if self_contained {
                &opts.post_link_objects_self_contained
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
                let escaped_output = escape_linker_output(&output, flavor);
                // FIXME: Add UI tests for this error.
                let err = errors::LinkingFailed {
                    linker_path: &linker_path,
                    exit_status: prog.status,
                    command: &cmd,
                    escaped_output,
                };
                sess.diagnostic().emit_err(err);
                // If MSVC's `link.exe` was expected but the return code
                // is not a Microsoft LNK error then suggest a way to fix or
                // install the Visual Studio build tools.
                if let Some(code) = prog.status.code() {
                    if sess.target.is_like_msvc
                        && flavor == LinkerFlavor::Msvc(Lld::No)
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

                        sess.emit_note(errors::LinkExeUnexpectedError);
                        if is_vs_installed && has_linker {
                            // the linker is broken
                            sess.emit_note(errors::RepairVSBuildTools);
                            sess.emit_note(errors::MissingCppBuildToolComponent);
                        } else if is_vs_installed {
                            // the linker is not installed
                            sess.emit_note(errors::SelectCppBuildToolWorkload);
                        } else {
                            // visual studio is not installed
                            sess.emit_note(errors::VisualStudioNotInstalled);
                        }
                    }
                }

                sess.abort_if_errors();
            }
            info!("linker stderr:\n{}", escape_string(&prog.stderr));
            info!("linker stdout:\n{}", escape_string(&prog.stdout));
        }
        Err(e) => {
            let linker_not_found = e.kind() == io::ErrorKind::NotFound;

            if linker_not_found {
                sess.emit_err(errors::LinkerNotFound { linker_path, error: e });
            } else {
                sess.emit_err(errors::UnableToExeLinker {
                    linker_path,
                    error: e,
                    command_formatted: format!("{:?}", &cmd),
                });
            }

            if sess.target.is_like_msvc && linker_not_found {
                sess.emit_note(errors::MsvcMissingLinker);
                sess.emit_note(errors::CheckInstalledVisualStudio);
                sess.emit_note(errors::InsufficientVSCodeProduct);
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
                        sess.emit_warning(errors::ProcessingDymutilFailed {
                            status: prog.status,
                            output: escape_string(&output),
                        });
                    }
                }
                Err(error) => sess.emit_fatal(errors::UnableToRunDsymutil { error }),
            }
        }

        // On MSVC packed debug information is produced by the linker itself so
        // there's no need to do anything else here.
        SplitDebuginfo::Packed if sess.target.is_like_windows => {}

        // ... and otherwise we're processing a `*.dwp` packed dwarf file.
        //
        // We cannot rely on the .o paths in the executable because they may have been
        // remapped by --remap-path-prefix and therefore invalid, so we need to provide
        // the .o/.dwo paths explicitly.
        SplitDebuginfo::Packed => link_dwarf_object(sess, codegen_results, out_filename),
    }

    let strip = strip_value(sess);

    if sess.target.is_like_osx {
        match (strip, crate_type) {
            (Strip::Debuginfo, _) => {
                strip_symbols_with_external_utility(sess, "strip", &out_filename, Some("-S"))
            }
            // Per the manpage, `-x` is the maximum safe strip level for dynamic libraries. (#93988)
            (Strip::Symbols, CrateType::Dylib | CrateType::Cdylib | CrateType::ProcMacro) => {
                strip_symbols_with_external_utility(sess, "strip", &out_filename, Some("-x"))
            }
            (Strip::Symbols, _) => {
                strip_symbols_with_external_utility(sess, "strip", &out_filename, None)
            }
            (Strip::None, _) => {}
        }
    }

    if sess.target.os == "illumos" {
        // Many illumos systems will have both the native 'strip' utility and
        // the GNU one. Use the native version explicitly and do not rely on
        // what's in the path.
        let stripcmd = "/usr/bin/strip";
        match strip {
            // Always preserve the symbol table (-x).
            Strip::Debuginfo => {
                strip_symbols_with_external_utility(sess, stripcmd, &out_filename, Some("-x"))
            }
            // Strip::Symbols is handled via the --strip-all linker option.
            Strip::Symbols => {}
            Strip::None => {}
        }
    }

    Ok(())
}

// Temporarily support both -Z strip and -C strip
fn strip_value(sess: &Session) -> Strip {
    match (sess.opts.unstable_opts.strip, sess.opts.cg.strip) {
        (s, Strip::None) => s,
        (_, s) => s,
    }
}

fn strip_symbols_with_external_utility<'a>(
    sess: &'a Session,
    util: &str,
    out_filename: &Path,
    option: Option<&str>,
) {
    let mut cmd = Command::new(util);
    if let Some(option) = option {
        cmd.arg(option);
    }
    let prog = cmd.arg(out_filename).output();
    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                sess.emit_warning(errors::StrippingDebugInfoFailed {
                    util,
                    status: prog.status,
                    output: escape_string(&output),
                });
            }
        }
        Err(error) => sess.emit_fatal(errors::UnableToRun { util, error }),
    }
}

fn escape_string(s: &[u8]) -> String {
    match str::from_utf8(s) {
        Ok(s) => s.to_owned(),
        Err(_) => format!("Non-UTF-8 output: {}", s.escape_ascii()),
    }
}

#[cfg(not(windows))]
fn escape_linker_output(s: &[u8], _flavour: LinkerFlavor) -> String {
    escape_string(s)
}

/// If the output of the msvc linker is not UTF-8 and the host is Windows,
/// then try to convert the string from the OEM encoding.
#[cfg(windows)]
fn escape_linker_output(s: &[u8], flavour: LinkerFlavor) -> String {
    // This only applies to the actual MSVC linker.
    if flavour != LinkerFlavor::Msvc(Lld::No) {
        return escape_string(s);
    }
    match str::from_utf8(s) {
        Ok(s) => return s.to_owned(),
        Err(_) => match win::locale_byte_str_to_string(s, win::oem_code_page()) {
            Some(s) => s,
            // The string is not UTF-8 and isn't valid for the OEM code page
            None => format!("Non-UTF-8 output: {}", s.escape_ascii()),
        },
    }
}

/// Wrappers around the Windows API.
#[cfg(windows)]
mod win {
    use windows::Win32::Globalization::{
        GetLocaleInfoEx, MultiByteToWideChar, CP_OEMCP, LOCALE_IUSEUTF8LEGACYOEMCP,
        LOCALE_NAME_SYSTEM_DEFAULT, LOCALE_RETURN_NUMBER, MB_ERR_INVALID_CHARS,
    };

    /// Get the Windows system OEM code page. This is most notably the code page
    /// used for link.exe's output.
    pub fn oem_code_page() -> u32 {
        unsafe {
            let mut cp: u32 = 0;
            // We're using the `LOCALE_RETURN_NUMBER` flag to return a u32.
            // But the API requires us to pass the data as though it's a [u16] string.
            let len = std::mem::size_of::<u32>() / std::mem::size_of::<u16>();
            let data = std::slice::from_raw_parts_mut(&mut cp as *mut u32 as *mut u16, len);
            let len_written = GetLocaleInfoEx(
                LOCALE_NAME_SYSTEM_DEFAULT,
                LOCALE_IUSEUTF8LEGACYOEMCP | LOCALE_RETURN_NUMBER,
                Some(data),
            );
            if len_written as usize == len { cp } else { CP_OEMCP }
        }
    }
    /// Try to convert a multi-byte string to a UTF-8 string using the given code page
    /// The string does not need to be null terminated.
    ///
    /// This is implemented as a wrapper around `MultiByteToWideChar`.
    /// See <https://learn.microsoft.com/en-us/windows/win32/api/stringapiset/nf-stringapiset-multibytetowidechar>
    ///
    /// It will fail if the multi-byte string is longer than `i32::MAX` or if it contains
    /// any invalid bytes for the expected encoding.
    pub fn locale_byte_str_to_string(s: &[u8], code_page: u32) -> Option<String> {
        // `MultiByteToWideChar` requires a length to be a "positive integer".
        if s.len() > isize::MAX as usize {
            return None;
        }
        // Error if the string is not valid for the expected code page.
        let flags = MB_ERR_INVALID_CHARS;
        // Call MultiByteToWideChar twice.
        // First to calculate the length then to convert the string.
        let mut len = unsafe { MultiByteToWideChar(code_page, flags, s, None) };
        if len > 0 {
            let mut utf16 = vec![0; len as usize];
            len = unsafe { MultiByteToWideChar(code_page, flags, s, Some(&mut utf16)) };
            if len > 0 {
                return utf16.get(..len as usize).map(String::from_utf16_lossy);
            }
        }
        None
    }
}

fn add_sanitizer_libraries(sess: &Session, crate_type: CrateType, linker: &mut dyn Linker) {
    // On macOS the runtimes are distributed as dylibs which should be linked to
    // both executables and dynamic shared objects. Everywhere else the runtimes
    // are currently distributed as static libraries which should be linked to
    // executables only.
    let needs_runtime = !sess.target.is_like_android
        && match crate_type {
            CrateType::Executable => true,
            CrateType::Dylib | CrateType::Cdylib | CrateType::ProcMacro => sess.target.is_like_osx,
            CrateType::Rlib | CrateType::Staticlib => false,
        };

    if !needs_runtime {
        return;
    }

    let sanitizer = sess.opts.unstable_opts.sanitizer;
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
    if sanitizer.contains(SanitizerSet::SAFESTACK) {
        link_sanitizer_runtime(sess, linker, "safestack");
    }
}

fn link_sanitizer_runtime(sess: &Session, linker: &mut dyn Linker, name: &str) {
    fn find_sanitizer_runtime(sess: &Session, filename: &str) -> PathBuf {
        let session_tlib =
            filesearch::make_target_lib_path(&sess.sysroot, sess.opts.target_triple.triple());
        let path = session_tlib.join(filename);
        if path.exists() {
            return session_tlib;
        } else {
            let default_sysroot =
                filesearch::get_or_default_sysroot().expect("Failed finding sysroot");
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
        linker.link_dylib(&filename, false, true);
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

/// This functions tries to determine the appropriate linker (and corresponding LinkerFlavor) to use
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
                    LinkerFlavor::Gnu(Cc::Yes, _)
                    | LinkerFlavor::Darwin(Cc::Yes, _)
                    | LinkerFlavor::WasmLld(Cc::Yes)
                    | LinkerFlavor::Unix(Cc::Yes) => {
                        if cfg!(any(target_os = "solaris", target_os = "illumos")) {
                            // On historical Solaris systems, "cc" may have
                            // been Sun Studio, which is not flag-compatible
                            // with "gcc". This history casts a long shadow,
                            // and many modern illumos distributions today
                            // ship GCC as "gcc" without also making it
                            // available as "cc".
                            "gcc"
                        } else {
                            "cc"
                        }
                    }
                    LinkerFlavor::Gnu(_, Lld::Yes)
                    | LinkerFlavor::Darwin(_, Lld::Yes)
                    | LinkerFlavor::WasmLld(..)
                    | LinkerFlavor::Msvc(Lld::Yes) => "lld",
                    LinkerFlavor::Gnu(..) | LinkerFlavor::Darwin(..) | LinkerFlavor::Unix(..) => {
                        "ld"
                    }
                    LinkerFlavor::Msvc(..) => "link.exe",
                    LinkerFlavor::EmCc => {
                        if cfg!(windows) {
                            "emcc.bat"
                        } else {
                            "emcc"
                        }
                    }
                    LinkerFlavor::Bpf => "bpf-linker",
                    LinkerFlavor::Ptx => "rust-ptx-linker",
                }),
                flavor,
            )),
            (Some(linker), None) => {
                let stem = linker.file_stem().and_then(|stem| stem.to_str()).unwrap_or_else(|| {
                    sess.emit_fatal(errors::LinkerFileStem);
                });
                let flavor = sess.target.linker_flavor.with_linker_hints(stem);
                Some((linker, flavor))
            }
            (None, None) => None,
        }
    }

    // linker and linker flavor specified via command line have precedence over what the target
    // specification specifies
    let linker_flavor =
        sess.opts.cg.linker_flavor.map(|flavor| sess.target.linker_flavor.with_cli_hints(flavor));
    if let Some(ret) = infer_from(sess, sess.opts.cg.linker.clone(), linker_flavor) {
        return ret;
    }

    if let Some(ret) = infer_from(
        sess,
        sess.target.linker.as_deref().map(PathBuf::from),
        Some(sess.target.linker_flavor),
    ) {
        return ret;
    }

    bug!("Not enough information provided to determine how to invoke the linker");
}

/// Returns a pair of boolean indicating whether we should preserve the object and
/// dwarf object files on the filesystem for their debug information. This is often
/// useful with split-dwarf like schemes.
fn preserve_objects_for_their_debuginfo(sess: &Session) -> (bool, bool) {
    // If the objects don't have debuginfo there's nothing to preserve.
    if sess.opts.debuginfo == config::DebugInfo::None {
        return (false, false);
    }

    match (sess.split_debuginfo(), sess.opts.unstable_opts.split_dwarf_kind) {
        // If there is no split debuginfo then do not preserve objects.
        (SplitDebuginfo::Off, _) => (false, false),
        // If there is packed split debuginfo, then the debuginfo in the objects
        // has been packaged and the objects can be deleted.
        (SplitDebuginfo::Packed, _) => (false, false),
        // If there is unpacked split debuginfo and the current target can not use
        // split dwarf, then keep objects.
        (SplitDebuginfo::Unpacked, _) if !sess.target_can_use_split_dwarf() => (true, false),
        // If there is unpacked split debuginfo and the target can use split dwarf, then
        // keep the object containing that debuginfo (whether that is an object file or
        // dwarf object file depends on the split dwarf kind).
        (SplitDebuginfo::Unpacked, SplitDwarfKind::Single) => (true, false),
        (SplitDebuginfo::Unpacked, SplitDwarfKind::Split) => (false, true),
    }
}

fn archive_search_paths(sess: &Session) -> Vec<PathBuf> {
    sess.target_filesearch(PathKind::Native).search_path_dirs()
}

#[derive(PartialEq)]
enum RlibFlavor {
    Normal,
    StaticlibBase,
}

fn print_native_static_libs(
    sess: &Session,
    all_native_libs: &[NativeLib],
    all_rust_dylibs: &[&Path],
) {
    let mut lib_args: Vec<_> = all_native_libs
        .iter()
        .filter(|l| relevant_lib(sess, l))
        .filter_map(|lib| {
            let name = lib.name;
            match lib.kind {
                NativeLibKind::Static { bundle: Some(false), .. }
                | NativeLibKind::Dylib { .. }
                | NativeLibKind::Unspecified => {
                    let verbatim = lib.verbatim;
                    if sess.target.is_like_msvc {
                        Some(format!("{}{}", name, if verbatim { "" } else { ".lib" }))
                    } else if sess.target.linker_flavor.is_gnu() {
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
                | NativeLibKind::LinkArg
                | NativeLibKind::WasmImportModule
                | NativeLibKind::RawDylib => None,
            }
        })
        .collect();
    for path in all_rust_dylibs {
        // FIXME deduplicate with add_dynamic_crate

        // Just need to tell the linker about where the library lives and
        // what its name is
        let parent = path.parent();
        if let Some(dir) = parent {
            let dir = fix_windows_verbatim_for_gcc(dir);
            if sess.target.is_like_msvc {
                let mut arg = String::from("/LIBPATH:");
                arg.push_str(&dir.display().to_string());
                lib_args.push(arg);
            } else {
                lib_args.push("-L".to_owned());
                lib_args.push(dir.display().to_string());
            }
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        // Convert library file-stem into a cc -l argument.
        let prefix = if stem.starts_with("lib") && !sess.target.is_like_windows { 3 } else { 0 };
        let lib = &stem[prefix..];
        let path = parent.unwrap_or_else(|| Path::new(""));
        if sess.target.is_like_msvc {
            // When producing a dll, the MSVC linker may not actually emit a
            // `foo.lib` file if the dll doesn't actually export any symbols, so we
            // check to see if the file is there and just omit linking to it if it's
            // not present.
            let name = format!("{}.dll.lib", lib);
            if path.join(&name).exists() {
                lib_args.push(name);
            }
        } else {
            lib_args.push(format!("-l{}", lib));
        }
    }
    if !lib_args.is_empty() {
        sess.emit_note(errors::StaticLibraryNativeArtifacts);
        // Prefix for greppability
        // Note: This must not be translated as tools are allowed to depend on this exact string.
        sess.note_without_error(format!("native-static-libs: {}", &lib_args.join(" ")));
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

/// Various toolchain components used during linking are used from rustc distribution
/// instead of being found somewhere on the host system.
/// We only provide such support for a very limited number of targets.
fn self_contained(sess: &Session, crate_type: CrateType) -> bool {
    if let Some(self_contained) = sess.opts.cg.link_self_contained.explicitly_set {
        if sess.target.link_self_contained == LinkSelfContainedDefault::False {
            sess.emit_err(errors::UnsupportedLinkSelfContained);
        }
        return self_contained;
    }

    match sess.target.link_self_contained {
        LinkSelfContainedDefault::False => false,
        LinkSelfContainedDefault::True => true,
        // FIXME: Find a better heuristic for "native musl toolchain is available",
        // based on host and linker path, for example.
        // (https://github.com/rust-lang/rust/pull/71769#issuecomment-626330237).
        LinkSelfContainedDefault::Musl => sess.crt_static(Some(crate_type)),
        LinkSelfContainedDefault::Mingw => {
            sess.host == sess.target
                && sess.target.vendor != "uwp"
                && detect_self_contained_mingw(&sess)
        }
    }
}

/// Add pre-link object files defined by the target spec.
fn add_pre_link_objects(
    cmd: &mut dyn Linker,
    sess: &Session,
    flavor: LinkerFlavor,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
) {
    // FIXME: we are currently missing some infra here (per-linker-flavor CRT objects),
    // so Fuchsia has to be special-cased.
    let opts = &sess.target;
    let empty = Default::default();
    let objects = if self_contained {
        &opts.pre_link_objects_self_contained
    } else if !(sess.target.os == "fuchsia" && matches!(flavor, LinkerFlavor::Gnu(Cc::Yes, _))) {
        &opts.pre_link_objects
    } else {
        &empty
    };
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
    let objects = if self_contained {
        &sess.target.post_link_objects_self_contained
    } else {
        &sess.target.post_link_objects
    };
    for obj in objects.get(&link_output_kind).iter().copied().flatten() {
        cmd.add_object(&get_object_file_path(sess, obj, self_contained));
    }
}

/// Add arbitrary "pre-link" args defined by the target spec or from command line.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_pre_link_args(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    if let Some(args) = sess.target.pre_link_args.get(&flavor) {
        cmd.args(args.iter().map(Deref::deref));
    }
    cmd.args(&sess.opts.unstable_opts.pre_link_args);
}

/// Add a link script embedded in the target, if applicable.
fn add_link_script(cmd: &mut dyn Linker, sess: &Session, tmpdir: &Path, crate_type: CrateType) {
    match (crate_type, &sess.target.link_script) {
        (CrateType::Cdylib | CrateType::Executable, Some(script)) => {
            if !sess.target.linker_flavor.is_gnu() {
                sess.emit_fatal(errors::LinkScriptUnavailable);
            }

            let file_name = ["rustc", &sess.target.llvm_target, "linkfile.ld"].join("-");

            let path = tmpdir.join(file_name);
            if let Err(error) = fs::write(&path, script.as_ref()) {
                sess.emit_fatal(errors::LinkScriptWriteFailure { path, error });
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
            cmd.args(args.iter().map(Deref::deref));
        }
    } else {
        if let Some(args) = sess.target.late_link_args_static.get(&flavor) {
            cmd.args(args.iter().map(Deref::deref));
        }
    }
    if let Some(args) = sess.target.late_link_args.get(&flavor) {
        cmd.args(args.iter().map(Deref::deref));
    }
}

/// Add arbitrary "post-link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_post_link_args(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    if let Some(args) = sess.target.post_link_args.get(&flavor) {
        cmd.args(args.iter().map(Deref::deref));
    }
}

/// Add a synthetic object file that contains reference to all symbols that we want to expose to
/// the linker.
///
/// Background: we implement rlibs as static library (archives). Linkers treat archives
/// differently from object files: all object files participate in linking, while archives will
/// only participate in linking if they can satisfy at least one undefined reference (version
/// scripts doesn't count). This causes `#[no_mangle]` or `#[used]` items to be ignored by the
/// linker, and since they never participate in the linking, using `KEEP` in the linker scripts
/// can't keep them either. This causes #47384.
///
/// To keep them around, we could use `--whole-archive` and equivalents to force rlib to
/// participate in linking like object files, but this proves to be expensive (#93791). Therefore
/// we instead just introduce an undefined reference to them. This could be done by `-u` command
/// line option to the linker or `EXTERN(...)` in linker scripts, however they does not only
/// introduce an undefined reference, but also make them the GC roots, preventing `--gc-sections`
/// from removing them, and this is especially problematic for embedded programming where every
/// byte counts.
///
/// This method creates a synthetic object file, which contains undefined references to all symbols
/// that are necessary for the linking. They are only present in symbol table but not actually
/// used in any sections, so the linker will therefore pick relevant rlibs for linking, but
/// unused `#[no_mangle]` or `#[used]` can still be discard by GC sections.
///
/// There's a few internal crates in the standard library (aka libcore and
/// libstd) which actually have a circular dependence upon one another. This
/// currently arises through "weak lang items" where libcore requires things
/// like `rust_begin_unwind` but libstd ends up defining it. To get this
/// circular dependence to work correctly we declare some of these things
/// in this synthetic object.
fn add_linked_symbol_object(
    cmd: &mut dyn Linker,
    sess: &Session,
    tmpdir: &Path,
    symbols: &[(String, SymbolExportKind)],
) {
    if symbols.is_empty() {
        return;
    }

    let Some(mut file) = super::metadata::create_object_file(sess) else {
        return;
    };

    // NOTE(nbdd0121): MSVC will hang if the input object file contains no sections,
    // so add an empty section.
    if file.format() == object::BinaryFormat::Coff {
        file.add_section(Vec::new(), ".text".into(), object::SectionKind::Text);

        // We handle the name decoration of COFF targets in `symbol_export.rs`, so disable the
        // default mangler in `object` crate.
        file.set_mangling(object::write::Mangling::None);

        // Add feature flags to the object file. On MSVC this is optional but LLD will complain if
        // not present.
        let mut feature = 0;

        if file.architecture() == object::Architecture::I386 {
            // Indicate that all SEH handlers are registered in .sxdata section.
            // We don't have generate any code, so we don't need .sxdata section but LLD still
            // expects us to set this bit (see #96498).
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
    }

    for (sym, kind) in symbols.iter() {
        file.add_symbol(object::write::Symbol {
            name: sym.clone().into(),
            value: 0,
            size: 0,
            kind: match kind {
                SymbolExportKind::Text => object::SymbolKind::Text,
                SymbolExportKind::Data => object::SymbolKind::Data,
                SymbolExportKind::Tls => object::SymbolKind::Tls,
            },
            scope: object::SymbolScope::Unknown,
            weak: false,
            section: object::write::SymbolSection::Undefined,
            flags: object::SymbolFlags::None,
        });
    }

    let path = tmpdir.join("symbols.o");
    let result = std::fs::write(&path, file.write().unwrap());
    if let Err(error) = result {
        sess.emit_fatal(errors::FailedToWrite { path, error });
    }
    cmd.add_object(&path);
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
    match sess.opts.unstable_opts.relro_level.unwrap_or(sess.target.relro_level) {
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
            linker_is_gnu: sess.target.linker_flavor.is_gnu(),
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
fn linker_with_args<'a>(
    path: &Path,
    flavor: LinkerFlavor,
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    crate_type: CrateType,
    tmpdir: &Path,
    out_filename: &Path,
    codegen_results: &CodegenResults,
) -> Result<Command, ErrorGuaranteed> {
    let self_contained = self_contained(sess, crate_type);
    let cmd = &mut *super::linker::get_linker(
        sess,
        path,
        flavor,
        self_contained,
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
    add_pre_link_objects(cmd, sess, flavor, link_output_kind, self_contained);

    add_linked_symbol_object(
        cmd,
        sess,
        tmpdir,
        &codegen_results.crate_info.linked_symbols[&crate_type],
    );

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

    // Local native libraries of all kinds.
    add_local_native_libraries(
        cmd,
        sess,
        archive_builder_builder,
        codegen_results,
        tmpdir,
        link_output_kind,
    );

    // Upstream rust crates and their non-dynamic native libraries.
    add_upstream_rust_crates(
        cmd,
        sess,
        archive_builder_builder,
        codegen_results,
        crate_type,
        tmpdir,
        link_output_kind,
    );

    // Dynamic native libraries from upstream crates.
    add_upstream_native_libraries(
        cmd,
        sess,
        archive_builder_builder,
        codegen_results,
        tmpdir,
        link_output_kind,
    );

    // Link with the import library generated for any raw-dylib functions.
    for (raw_dylib_name, raw_dylib_imports) in
        collate_raw_dylibs(sess, codegen_results.crate_info.used_libraries.iter())?
    {
        cmd.add_object(&archive_builder_builder.create_dll_import_lib(
            sess,
            &raw_dylib_name,
            &raw_dylib_imports,
            tmpdir,
            true,
        ));
    }
    // As with add_upstream_native_libraries, we need to add the upstream raw-dylib symbols in case
    // they are used within inlined functions or instantiated generic functions. We do this *after*
    // handling the raw-dylib symbols in the current crate to make sure that those are chosen first
    // by the linker.
    let (_, dependency_linkage) = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| *ty == crate_type)
        .expect("failed to find crate type in dependency format list");
    let native_libraries_from_nonstatics = codegen_results
        .crate_info
        .native_libraries
        .iter()
        .filter_map(|(cnum, libraries)| {
            (dependency_linkage[cnum.as_usize() - 1] != Linkage::Static).then_some(libraries)
        })
        .flatten();
    for (raw_dylib_name, raw_dylib_imports) in
        collate_raw_dylibs(sess, native_libraries_from_nonstatics)?
    {
        cmd.add_object(&archive_builder_builder.create_dll_import_lib(
            sess,
            &raw_dylib_name,
            &raw_dylib_imports,
            tmpdir,
            false,
        ));
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
        self_contained,
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
    add_post_link_objects(cmd, sess, link_output_kind, self_contained);

    // ------------ Late order-dependent options ------------

    // Doesn't really make sense.
    // FIXME: In practice built-in target specs use this for arbitrary order-independent options,
    // introduce a target spec option for order-independent linker options, migrate built-in specs
    // to it and remove the option.
    add_post_link_args(cmd, sess, flavor);

    Ok(cmd.take_cmd())
}

fn add_order_independent_options(
    cmd: &mut dyn Linker,
    sess: &Session,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
    flavor: LinkerFlavor,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
    out_filename: &Path,
    tmpdir: &Path,
) {
    // Take care of the flavors and CLI options requesting the `lld` linker.
    add_lld_args(cmd, sess, flavor);

    add_apple_sdk(cmd, sess, flavor);

    add_link_script(cmd, sess, tmpdir, crate_type);

    if sess.target.os == "fuchsia"
        && crate_type == CrateType::Executable
        && !matches!(flavor, LinkerFlavor::Gnu(Cc::Yes, _))
    {
        let prefix = if sess.opts.unstable_opts.sanitizer.contains(SanitizerSet::ADDRESS) {
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

    if self_contained {
        cmd.no_crt_objects();
    }

    if sess.target.os == "emscripten" {
        cmd.arg("-s");
        cmd.arg(if sess.panic_strategy() == PanicStrategy::Abort {
            "DISABLE_EXCEPTION_CATCHING=1"
        } else {
            "DISABLE_EXCEPTION_CATCHING=0"
        });
    }

    if flavor == LinkerFlavor::Ptx {
        // Provide the linker with fallback to internal `target-cpu`.
        cmd.arg("--fallback-arch");
        cmd.arg(&codegen_results.crate_info.target_cpu);
    } else if flavor == LinkerFlavor::Bpf {
        cmd.arg("--cpu");
        cmd.arg(&codegen_results.crate_info.target_cpu);
        if let Some(feat) = [sess.opts.cg.target_feature.as_str(), &sess.target.options.features]
            .into_iter()
            .find(|feat| !feat.is_empty())
        {
            cmd.arg("--cpu-features");
            cmd.arg(feat);
        }
    }

    cmd.linker_plugin_lto();

    add_library_search_dirs(cmd, sess, self_contained);

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
        if crate_type != CrateType::Executable || !sess.opts.unstable_opts.export_executable_symbols
        {
            cmd.gc_sections(keep_metadata);
        } else {
            cmd.no_gc_sections();
        }
    }

    cmd.set_output_kind(link_output_kind, out_filename);

    add_relro_args(cmd, sess);

    // Pass optimization flags down to the linker.
    cmd.optimize();

    // Gather the set of NatVis files, if any, and write them out to a temp directory.
    let natvis_visualizers = collect_natvis_visualizers(
        tmpdir,
        sess,
        &codegen_results.crate_info.local_crate_name,
        &codegen_results.crate_info.natvis_debugger_visualizers,
    );

    // Pass debuginfo, NatVis debugger visualizers and strip flags down to the linker.
    cmd.debuginfo(strip_value(sess), &natvis_visualizers);

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

// Write the NatVis debugger visualizer files for each crate to the temp directory and gather the file paths.
fn collect_natvis_visualizers(
    tmpdir: &Path,
    sess: &Session,
    crate_name: &Symbol,
    natvis_debugger_visualizers: &BTreeSet<DebuggerVisualizerFile>,
) -> Vec<PathBuf> {
    let mut visualizer_paths = Vec::with_capacity(natvis_debugger_visualizers.len());

    for (index, visualizer) in natvis_debugger_visualizers.iter().enumerate() {
        let visualizer_out_file = tmpdir.join(format!("{}-{}.natvis", crate_name.as_str(), index));

        match fs::write(&visualizer_out_file, &visualizer.src) {
            Ok(()) => {
                visualizer_paths.push(visualizer_out_file);
            }
            Err(error) => {
                sess.emit_warning(errors::UnableToWriteDebuggerVisualizer {
                    path: visualizer_out_file,
                    error,
                });
            }
        };
    }
    visualizer_paths
}

fn add_native_libs_from_crate(
    cmd: &mut dyn Linker,
    sess: &Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
    search_paths: &OnceCell<Vec<PathBuf>>,
    bundled_libs: &FxHashSet<Symbol>,
    cnum: CrateNum,
    link_static: bool,
    link_dynamic: bool,
    link_output_kind: LinkOutputKind,
) {
    if !sess.opts.unstable_opts.link_native_libraries {
        // If `-Zlink-native-libraries=false` is set, then the assumption is that an
        // external build system already has the native dependencies defined, and it
        // will provide them to the linker itself.
        return;
    }

    if link_static && cnum != LOCAL_CRATE && !bundled_libs.is_empty() {
        // If rlib contains native libs as archives, unpack them to tmpdir.
        let rlib = &codegen_results.crate_info.used_crate_source[&cnum].rlib.as_ref().unwrap().0;
        archive_builder_builder
            .extract_bundled_libs(rlib, tmpdir, &bundled_libs)
            .unwrap_or_else(|e| sess.emit_fatal(e));
    }

    let native_libs = match cnum {
        LOCAL_CRATE => &codegen_results.crate_info.used_libraries,
        _ => &codegen_results.crate_info.native_libraries[&cnum],
    };

    let mut last = (None, NativeLibKind::Unspecified, false);
    for lib in native_libs {
        if !relevant_lib(sess, lib) {
            continue;
        }

        // Skip if this library is the same as the last.
        last = if (Some(lib.name), lib.kind, lib.verbatim) == last {
            continue;
        } else {
            (Some(lib.name), lib.kind, lib.verbatim)
        };

        let name = lib.name.as_str();
        let verbatim = lib.verbatim;
        match lib.kind {
            NativeLibKind::Static { bundle, whole_archive } => {
                if link_static {
                    let bundle = bundle.unwrap_or(true);
                    let whole_archive = whole_archive == Some(true)
                        // Backward compatibility case: this can be a rlib (so `+whole-archive`
                        // cannot be added explicitly if necessary, see the error in `fn link_rlib`)
                        // compiled as an executable due to `--test`. Use whole-archive implicitly,
                        // like before the introduction of native lib modifiers.
                        || (whole_archive == None
                            && bundle
                            && cnum == LOCAL_CRATE
                            && sess.is_test_crate());

                    if bundle && cnum != LOCAL_CRATE {
                        if let Some(filename) = lib.filename {
                            // If rlib contains native libs as archives, they are unpacked to tmpdir.
                            let path = tmpdir.join(filename.as_str());
                            if whole_archive {
                                cmd.link_whole_rlib(&path);
                            } else {
                                cmd.link_rlib(&path);
                            }
                        }
                    } else {
                        if whole_archive {
                            cmd.link_whole_staticlib(
                                name,
                                verbatim,
                                &search_paths.get_or_init(|| archive_search_paths(sess)),
                            );
                        } else {
                            cmd.link_staticlib(name, verbatim)
                        }
                    }
                }
            }
            NativeLibKind::Dylib { as_needed } => {
                if link_dynamic {
                    cmd.link_dylib(name, verbatim, as_needed.unwrap_or(true))
                }
            }
            NativeLibKind::Unspecified => {
                // If we are generating a static binary, prefer static library when the
                // link kind is unspecified.
                if !link_output_kind.can_link_dylib() && !sess.target.crt_static_allows_dylibs {
                    if link_static {
                        cmd.link_staticlib(name, verbatim)
                    }
                } else {
                    if link_dynamic {
                        cmd.link_dylib(name, verbatim, true);
                    }
                }
            }
            NativeLibKind::Framework { as_needed } => {
                if link_dynamic {
                    cmd.link_framework(name, as_needed.unwrap_or(true))
                }
            }
            NativeLibKind::RawDylib => {
                // Handled separately in `linker_with_args`.
            }
            NativeLibKind::WasmImportModule => {}
            NativeLibKind::LinkArg => {
                if link_static {
                    cmd.arg(name);
                }
            }
        }
    }
}

fn add_local_native_libraries(
    cmd: &mut dyn Linker,
    sess: &Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
    link_output_kind: LinkOutputKind,
) {
    if sess.opts.unstable_opts.link_native_libraries {
        // User-supplied library search paths (-L on the command line). These are the same paths
        // used to find Rust crates, so some of them may have been added already by the previous
        // crate linking code. This only allows them to be found at compile time so it is still
        // entirely up to outside forces to make sure that library can be found at runtime.
        for search_path in sess.target_filesearch(PathKind::All).search_paths() {
            match search_path.kind {
                PathKind::Framework => cmd.framework_path(&search_path.dir),
                _ => cmd.include_path(&fix_windows_verbatim_for_gcc(&search_path.dir)),
            }
        }
    }

    let search_paths = OnceCell::new();
    // All static and dynamic native library dependencies are linked to the local crate.
    let link_static = true;
    let link_dynamic = true;
    add_native_libs_from_crate(
        cmd,
        sess,
        archive_builder_builder,
        codegen_results,
        tmpdir,
        &search_paths,
        &Default::default(),
        LOCAL_CRATE,
        link_static,
        link_dynamic,
        link_output_kind,
    );
}

fn add_upstream_rust_crates<'a>(
    cmd: &mut dyn Linker,
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    crate_type: CrateType,
    tmpdir: &Path,
    link_output_kind: LinkOutputKind,
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

    let search_paths = OnceCell::new();
    for &cnum in &codegen_results.crate_info.used_crates {
        // We may not pass all crates through to the linker. Some crates may appear statically in
        // an existing dylib, meaning we'll pick up all the symbols from the dylib.
        // We must always link crates `compiler_builtins` and `profiler_builtins` statically.
        // Even if they were already included into a dylib
        // (e.g. `libstd` when `-C prefer-dynamic` is used).
        // FIXME: `dependency_formats` can report `profiler_builtins` as `NotLinked` for some
        // reason, it shouldn't do that because `profiler_builtins` should indeed be linked.
        let linkage = data[cnum.as_usize() - 1];
        let link_static_crate = linkage == Linkage::Static
            || (linkage == Linkage::IncludedFromDylib || linkage == Linkage::NotLinked)
                && (codegen_results.crate_info.compiler_builtins == Some(cnum)
                    || codegen_results.crate_info.profiler_runtime == Some(cnum));

        let mut bundled_libs = Default::default();
        match linkage {
            Linkage::Static | Linkage::IncludedFromDylib | Linkage::NotLinked => {
                if link_static_crate {
                    bundled_libs = codegen_results.crate_info.native_libraries[&cnum]
                        .iter()
                        .filter_map(|lib| lib.filename)
                        .collect();
                    add_static_crate(
                        cmd,
                        sess,
                        archive_builder_builder,
                        codegen_results,
                        tmpdir,
                        cnum,
                        &bundled_libs,
                    );
                }
            }
            Linkage::Dynamic => {
                let src = &codegen_results.crate_info.used_crate_source[&cnum];
                add_dynamic_crate(cmd, sess, &src.dylib.as_ref().unwrap().0);
            }
        }

        // Static libraries are linked for a subset of linked upstream crates.
        // 1. If the upstream crate is a directly linked rlib then we must link the native library
        // because the rlib is just an archive.
        // 2. If the upstream crate is a dylib or a rlib linked through dylib, then we do not link
        // the native library because it is already linked into the dylib, and even if
        // inline/const/generic functions from the dylib can refer to symbols from the native
        // library, those symbols should be exported and available from the dylib anyway.
        // 3. Libraries bundled into `(compiler,profiler)_builtins` are special, see above.
        let link_static = link_static_crate;
        // Dynamic libraries are not linked here, see the FIXME in `add_upstream_native_libraries`.
        let link_dynamic = false;
        add_native_libs_from_crate(
            cmd,
            sess,
            archive_builder_builder,
            codegen_results,
            tmpdir,
            &search_paths,
            &bundled_libs,
            cnum,
            link_static,
            link_dynamic,
            link_output_kind,
        );
    }
}

fn add_upstream_native_libraries(
    cmd: &mut dyn Linker,
    sess: &Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
    link_output_kind: LinkOutputKind,
) {
    let search_path = OnceCell::new();
    for &cnum in &codegen_results.crate_info.used_crates {
        // Static libraries are not linked here, they are linked in `add_upstream_rust_crates`.
        // FIXME: Merge this function to `add_upstream_rust_crates` so that all native libraries
        // are linked together with their respective upstream crates, and in their originally
        // specified order. This is slightly breaking due to our use of `--as-needed` (see crater
        // results in https://github.com/rust-lang/rust/pull/102832#issuecomment-1279772306).
        let link_static = false;
        // Dynamic libraries are linked for all linked upstream crates.
        // 1. If the upstream crate is a directly linked rlib then we must link the native library
        // because the rlib is just an archive.
        // 2. If the upstream crate is a dylib or a rlib linked through dylib, then we have to link
        // the native library too because inline/const/generic functions from the dylib can refer
        // to symbols from the native library, so the native library providing those symbols should
        // be available when linking our final binary.
        let link_dynamic = true;
        add_native_libs_from_crate(
            cmd,
            sess,
            archive_builder_builder,
            codegen_results,
            tmpdir,
            &search_path,
            &Default::default(),
            cnum,
            link_static,
            link_dynamic,
            link_output_kind,
        );
    }
}

// Rehome lib paths (which exclude the library file name) that point into the sysroot lib directory
// to be relative to the sysroot directory, which may be a relative path specified by the user.
//
// If the sysroot is a relative path, and the sysroot libs are specified as an absolute path, the
// linker command line can be non-deterministic due to the paths including the current working
// directory. The linker command line needs to be deterministic since it appears inside the PDB
// file generated by the MSVC linker. See https://github.com/rust-lang/rust/issues/112586.
//
// The returned path will always have `fix_windows_verbatim_for_gcc()` applied to it.
fn rehome_sysroot_lib_dir<'a>(sess: &'a Session, lib_dir: &Path) -> PathBuf {
    let sysroot_lib_path = sess.target_filesearch(PathKind::All).get_lib_path();
    let canonical_sysroot_lib_path =
        { try_canonicalize(&sysroot_lib_path).unwrap_or_else(|_| sysroot_lib_path.clone()) };

    let canonical_lib_dir = try_canonicalize(lib_dir).unwrap_or_else(|_| lib_dir.to_path_buf());
    if canonical_lib_dir == canonical_sysroot_lib_path {
        // This path, returned by `target_filesearch().get_lib_path()`, has
        // already had `fix_windows_verbatim_for_gcc()` applied if needed.
        sysroot_lib_path
    } else {
        fix_windows_verbatim_for_gcc(&lib_dir)
    }
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
fn add_static_crate<'a>(
    cmd: &mut dyn Linker,
    sess: &'a Session,
    archive_builder_builder: &dyn ArchiveBuilderBuilder,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
    cnum: CrateNum,
    bundled_lib_file_names: &FxHashSet<Symbol>,
) {
    let src = &codegen_results.crate_info.used_crate_source[&cnum];
    let cratepath = &src.rlib.as_ref().unwrap().0;

    let mut link_upstream = |path: &Path| {
        let rlib_path = if let Some(dir) = path.parent() {
            let file_name = path.file_name().expect("rlib path has no file name path component");
            rehome_sysroot_lib_dir(sess, &dir).join(file_name)
        } else {
            fix_windows_verbatim_for_gcc(path)
        };
        cmd.link_rlib(&rlib_path);
    };

    if !are_upstream_rust_objects_already_included(sess)
        || ignored_for_lto(sess, &codegen_results.crate_info, cnum)
    {
        link_upstream(cratepath);
        return;
    }

    let dst = tmpdir.join(cratepath.file_name().unwrap());
    let name = cratepath.file_name().unwrap().to_str().unwrap();
    let name = &name[3..name.len() - 5]; // chop off lib/.rlib
    let bundled_lib_file_names = bundled_lib_file_names.clone();

    sess.prof.generic_activity_with_arg("link_altering_rlib", name).run(|| {
        let canonical_name = name.replace('-', "_");
        let upstream_rust_objects_already_included =
            are_upstream_rust_objects_already_included(sess);
        let is_builtins =
            sess.target.no_builtins || !codegen_results.crate_info.is_no_builtins.contains(&cnum);

        let mut archive = archive_builder_builder.new_archive_builder(sess);
        if let Err(error) = archive.add_archive(
            cratepath,
            Box::new(move |f| {
                if f == METADATA_FILENAME {
                    return true;
                }

                let canonical = f.replace('-', "_");

                let is_rust_object =
                    canonical.starts_with(&canonical_name) && looks_like_rust_object_file(&f);

                // If we're performing LTO and this is a rust-generated object
                // file, then we don't need the object file as it's part of the
                // LTO module. Note that `#![no_builtins]` is excluded from LTO,
                // though, so we let that object file slide.
                if upstream_rust_objects_already_included && is_rust_object && is_builtins {
                    return true;
                }

                // We skip native libraries because:
                // 1. This native libraries won't be used from the generated rlib,
                //    so we can throw them away to avoid the copying work.
                // 2. We can't allow it to be a single remaining entry in archive
                //    as some linkers may complain on that.
                if bundled_lib_file_names.contains(&Symbol::intern(f)) {
                    return true;
                }

                false
            }),
        ) {
            sess.emit_fatal(errors::RlibArchiveBuildFailure { error });
        }
        if archive.build(&dst) {
            link_upstream(&dst);
        }
    });
}

// Same thing as above, but for dynamic crates instead of static crates.
fn add_dynamic_crate(cmd: &mut dyn Linker, sess: &Session, cratepath: &Path) {
    // Just need to tell the linker about where the library lives and
    // what its name is
    let parent = cratepath.parent();
    if let Some(dir) = parent {
        cmd.include_path(&rehome_sysroot_lib_dir(sess, dir));
    }
    let stem = cratepath.file_stem().unwrap().to_str().unwrap();
    // Convert library file-stem into a cc -l argument.
    let prefix = if stem.starts_with("lib") && !sess.target.is_like_windows { 3 } else { 0 };
    cmd.link_rust_dylib(&stem[prefix..], parent.unwrap_or_else(|| Path::new("")));
}

fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => rustc_attr::cfg_matches(cfg, &sess.parse_sess, CRATE_NODE_ID, None),
        None => true,
    }
}

pub(crate) fn are_upstream_rust_objects_already_included(sess: &Session) -> bool {
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
        || !matches!(os.as_ref(), "ios" | "tvos" | "watchos" | "macos")
        || !matches!(flavor, LinkerFlavor::Darwin(..))
    {
        return;
    }

    if os == "macos" && !matches!(flavor, LinkerFlavor::Darwin(Cc::No, _)) {
        return;
    }

    let sdk_name = match (arch.as_ref(), os.as_ref()) {
        ("aarch64", "tvos") => "appletvos",
        ("x86_64", "tvos") => "appletvsimulator",
        ("arm", "ios") => "iphoneos",
        ("aarch64", "ios") if llvm_target.contains("macabi") => "macosx",
        ("aarch64", "ios") if llvm_target.ends_with("-simulator") => "iphonesimulator",
        ("aarch64", "ios") => "iphoneos",
        ("x86", "ios") => "iphonesimulator",
        ("x86_64", "ios") if llvm_target.contains("macabi") => "macosx",
        ("x86_64", "ios") => "iphonesimulator",
        ("x86_64", "watchos") => "watchsimulator",
        ("arm64_32", "watchos") => "watchos",
        ("aarch64", "watchos") if llvm_target.ends_with("-simulator") => "watchsimulator",
        ("aarch64", "watchos") => "watchos",
        ("arm", "watchos") => "watchos",
        (_, "macos") => "macosx",
        _ => {
            sess.emit_err(errors::UnsupportedArch { arch, os });
            return;
        }
    };
    let sdk_root = match get_apple_sdk_root(sdk_name) {
        Ok(s) => s,
        Err(e) => {
            sess.emit_err(e);
            return;
        }
    };

    match flavor {
        LinkerFlavor::Darwin(Cc::Yes, _) => {
            cmd.args(&["-isysroot", &sdk_root, "-Wl,-syslibroot", &sdk_root]);
        }
        LinkerFlavor::Darwin(Cc::No, _) => {
            cmd.args(&["-syslibroot", &sdk_root]);
        }
        _ => unreachable!(),
    }
}

fn get_apple_sdk_root(sdk_name: &str) -> Result<String, errors::AppleSdkRootError<'_>> {
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
            "watchos"
                if sdkroot.contains("WatchSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "watchsimulator"
                if sdkroot.contains("WatchOS.platform") || sdkroot.contains("MacOSX.platform") => {}
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
        Err(error) => Err(errors::AppleSdkRootError::SdkPath { sdk_name, error }),
    }
}

/// When using the linker flavors opting in to `lld`, or the unstable `-Zgcc-ld=lld` flag, add the
/// necessary paths and arguments to invoke it:
/// - when the self-contained linker flag is active: the build of `lld` distributed with rustc,
/// - or any `lld` available to `cc`.
fn add_lld_args(cmd: &mut dyn Linker, sess: &Session, flavor: LinkerFlavor) {
    let unstable_use_lld = sess.opts.unstable_opts.gcc_ld.is_some();
    debug!("add_lld_args requested, flavor: '{flavor:?}', `-Zgcc-ld=lld`: {unstable_use_lld}");

    // Sanity check: using the old unstable `-Zgcc-ld=lld` option requires a `cc`-using flavor.
    let flavor_uses_cc = flavor.uses_cc();
    if unstable_use_lld && !flavor_uses_cc {
        sess.emit_fatal(errors::OptionGccOnly);
    }

    // If the flavor doesn't use a C/C++ compiler to invoke the linker, or doesn't opt in to `lld`,
    // we don't need to do anything.
    let use_lld = flavor.uses_lld() || unstable_use_lld;
    if !flavor_uses_cc || !use_lld {
        return;
    }

    // 1. Implement the "self-contained" part of this feature by adding rustc distribution
    //    directories to the tool's search path.
    let self_contained_linker = sess.opts.cg.link_self_contained.linker() || unstable_use_lld;
    if self_contained_linker {
        for path in sess.get_tools_search_paths(false) {
            cmd.arg({
                let mut arg = OsString::from("-B");
                arg.push(path.join("gcc-ld"));
                arg
            });
        }
    }

    // 2. Implement the "linker flavor" part of this feature by asking `cc` to use some kind of
    //    `lld` as the linker.
    cmd.arg("-fuse-ld=lld");

    if !flavor.is_gnu() {
        // Tell clang to use a non-default LLD flavor.
        // Gcc doesn't understand the target option, but we currently assume
        // that gcc is not used for Apple and Wasm targets (#97402).
        //
        // Note that we don't want to do that by default on macOS: e.g. passing a
        // 10.7 target to LLVM works, but not to recent versions of clang/macOS, as
        // shown in issue #101653 and the discussion in PR #101792.
        //
        // It could be required in some cases of cross-compiling with
        // `-Zgcc-ld=lld`, but this is generally unspecified, and we don't know
        // which specific versions of clang, macOS SDK, host and target OS
        // combinations impact us here.
        //
        // So we do a simple first-approximation until we know more of what the
        // Apple targets require (and which would be handled prior to hitting this
        // `-Zgcc-ld=lld` codepath anyway), but the expectation is that until then
        // this should be manually passed if needed. We specify the target when
        // targeting a different linker flavor on macOS, and that's also always
        // the case when targeting WASM.
        if sess.target.linker_flavor != sess.host.linker_flavor {
            cmd.arg(format!("--target={}", sess.target.llvm_target));
        }
    }
}
