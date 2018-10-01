// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::wasm;
use cc::windows_registry;
use super::archive::{ArchiveBuilder, ArchiveConfig};
use super::bytecode::RLIB_BYTECODE_EXTENSION;
use super::linker::Linker;
use super::command::Command;
use super::rpath::RPathConfig;
use super::rpath;
use metadata::METADATA_FILENAME;
use rustc::session::config::{self, DebugInfo, OutputFilenames, OutputType, PrintRequest};
use rustc::session::config::{RUST_CGU_EXT, Lto};
use rustc::session::filesearch;
use rustc::session::search_paths::PathKind;
use rustc::session::Session;
use rustc::middle::cstore::{NativeLibrary, LibSource, NativeLibraryKind};
use rustc::middle::dependency_format::Linkage;
use {CodegenResults, CrateInfo};
use rustc::util::common::time;
use rustc_fs_util::fix_windows_verbatim_for_gcc;
use rustc::hir::def_id::CrateNum;
use tempfile::{Builder as TempFileBuilder, TempDir};
use rustc_target::spec::{PanicStrategy, RelroLevel, LinkerFlavor};
use rustc_data_structures::fx::FxHashSet;
use context::get_reloc_model;
use llvm;

use std::ascii;
use std::char;
use std::env;
use std::fmt;
use std::fs;
use std::io;
use std::iter;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::str;
use syntax::attr;

pub use rustc_codegen_utils::link::{find_crate_name, filename_for_input, default_output_for_target,
                                  invalid_output_for_target, out_filename, check_file_is_writeable,
                                  filename_for_metadata};

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
pub fn get_linker(sess: &Session, linker: &Path, flavor: LinkerFlavor) -> (PathBuf, Command) {
    let msvc_tool = windows_registry::find_tool(&sess.opts.target_triple.triple(), "link.exe");

    // If our linker looks like a batch script on Windows then to execute this
    // we'll need to spawn `cmd` explicitly. This is primarily done to handle
    // emscripten where the linker is `emcc.bat` and needs to be spawned as
    // `cmd /c emcc.bat ...`.
    //
    // This worked historically but is needed manually since #42436 (regression
    // was tagged as #42791) and some more info can be found on #44443 for
    // emscripten itself.
    let mut cmd = match linker.to_str() {
        Some(linker) if cfg!(windows) && linker.ends_with(".bat") => Command::bat_script(linker),
        _ => match flavor {
            LinkerFlavor::Lld(f) => Command::lld(linker, f),
            LinkerFlavor::Msvc
                if sess.opts.cg.linker.is_none() && sess.target.target.options.linker.is_none() =>
            {
                Command::new(msvc_tool.as_ref().map(|t| t.path()).unwrap_or(linker))
            },
            _ => Command::new(linker),
        }
    };

    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = sess.host_filesearch(PathKind::All)
                           .get_tools_search_paths();
    let mut msvc_changed_path = false;
    if sess.target.target.options.is_like_msvc {
        if let Some(ref tool) = msvc_tool {
            cmd.args(tool.args());
            for &(ref k, ref v) in tool.env() {
                if k == "PATH" {
                    new_path.extend(env::split_paths(v));
                    msvc_changed_path = true;
                } else {
                    cmd.env(k, v);
                }
            }
        }
    }

    if !msvc_changed_path {
        if let Some(path) = env::var_os("PATH") {
            new_path.extend(env::split_paths(&path));
        }
    }
    cmd.env("PATH", env::join_paths(new_path).unwrap());

    (linker.to_path_buf(), cmd)
}

pub fn remove(sess: &Session, path: &Path) {
    match fs::remove_file(path) {
        Ok(..) => {}
        Err(e) => {
            sess.err(&format!("failed to remove {}: {}",
                             path.display(),
                             e));
        }
    }
}

/// Perform the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub(crate) fn link_binary(sess: &Session,
                          codegen_results: &CodegenResults,
                          outputs: &OutputFilenames,
                          crate_name: &str) -> Vec<PathBuf> {
    let mut out_filenames = Vec::new();
    for &crate_type in sess.crate_types.borrow().iter() {
        // Ignore executable crates if we have -Z no-codegen, as they will error.
        let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
        if (sess.opts.debugging_opts.no_codegen || !sess.opts.output_types.should_codegen()) &&
           !output_metadata &&
           crate_type == config::CrateType::Executable {
            continue;
        }

        if invalid_output_for_target(sess, crate_type) {
           bug!("invalid output type `{:?}` for target os `{}`",
                crate_type, sess.opts.target_triple);
        }
        let mut out_files = link_binary_output(sess,
                                               codegen_results,
                                               crate_type,
                                               outputs,
                                               crate_name);
        out_filenames.append(&mut out_files);
    }

    // Remove the temporary object file and metadata if we aren't saving temps
    if !sess.opts.cg.save_temps {
        if sess.opts.output_types.should_codegen() &&
            !preserve_objects_for_their_debuginfo(sess)
        {
            for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
                remove(sess, obj);
            }
        }
        for obj in codegen_results.modules.iter().filter_map(|m| m.bytecode_compressed.as_ref()) {
            remove(sess, obj);
        }
        if let Some(ref obj) = codegen_results.metadata_module.object {
            remove(sess, obj);
        }
        if let Some(ref allocator) = codegen_results.allocator_module {
            if let Some(ref obj) = allocator.object {
                remove(sess, obj);
            }
            if let Some(ref bc) = allocator.bytecode_compressed {
                remove(sess, bc);
            }
        }
    }

    out_filenames
}

/// Returns a boolean indicating whether we should preserve the object files on
/// the filesystem for their debug information. This is often useful with
/// split-dwarf like schemes.
fn preserve_objects_for_their_debuginfo(sess: &Session) -> bool {
    // If the objects don't have debuginfo there's nothing to preserve.
    if sess.opts.debuginfo == DebugInfo::None {
        return false
    }

    // If we're only producing artifacts that are archives, no need to preserve
    // the objects as they're losslessly contained inside the archives.
    let output_linked = sess.crate_types.borrow()
        .iter()
        .any(|x| *x != config::CrateType::Rlib && *x != config::CrateType::Staticlib);
    if !output_linked {
        return false
    }

    // If we're on OSX then the equivalent of split dwarf is turned on by
    // default. The final executable won't actually have any debug information
    // except it'll have pointers to elsewhere. Historically we've always run
    // `dsymutil` to "link all the dwarf together" but this is actually sort of
    // a bummer for incremental compilation! (the whole point of split dwarf is
    // that you don't do this sort of dwarf link).
    //
    // Basically as a result this just means that if we're on OSX and we're
    // *not* running dsymutil then the object files are the only source of truth
    // for debug information, so we must preserve them.
    if sess.target.target.options.is_like_osx {
        match sess.opts.debugging_opts.run_dsymutil {
            // dsymutil is not being run, preserve objects
            Some(false) => return true,

            // dsymutil is being run, no need to preserve the objects
            Some(true) => return false,

            // The default historical behavior was to always run dsymutil, so
            // we're preserving that temporarily, but we're likely to switch the
            // default soon.
            None => return false,
        }
    }

    false
}

pub(crate) fn each_linked_rlib(sess: &Session,
                               info: &CrateInfo,
                               f: &mut dyn FnMut(CrateNum, &Path)) -> Result<(), String> {
    let crates = info.used_crates_static.iter();
    let fmts = sess.dependency_formats.borrow();
    let fmts = fmts.get(&config::CrateType::Executable)
                   .or_else(|| fmts.get(&config::CrateType::Staticlib))
                   .or_else(|| fmts.get(&config::CrateType::Cdylib))
                   .or_else(|| fmts.get(&config::CrateType::ProcMacro));
    let fmts = match fmts {
        Some(f) => f,
        None => return Err("could not find formats for rlibs".to_string())
    };
    for &(cnum, ref path) in crates {
        match fmts.get(cnum.as_usize() - 1) {
            Some(&Linkage::NotLinked) |
            Some(&Linkage::IncludedFromDylib) => continue,
            Some(_) => {}
            None => return Err("could not find formats for rlibs".to_string())
        }
        let name = &info.crate_name[&cnum];
        let path = match *path {
            LibSource::Some(ref p) => p,
            LibSource::MetadataOnly => {
                return Err(format!("could not find rlib for: `{}`, found rmeta (metadata) file",
                                   name))
            }
            LibSource::None => {
                return Err(format!("could not find rlib for: `{}`", name))
            }
        };
        f(cnum, &path);
    }
    Ok(())
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
pub(crate) fn ignored_for_lto(sess: &Session, info: &CrateInfo, cnum: CrateNum) -> bool {
    // If our target enables builtin function lowering in LLVM then the
    // crates providing these functions don't participate in LTO (e.g.
    // no_builtins or compiler builtins crates).
    !sess.target.target.options.no_builtins &&
        (info.is_no_builtins.contains(&cnum) || info.compiler_builtins == Some(cnum))
}

fn link_binary_output(sess: &Session,
                      codegen_results: &CodegenResults,
                      crate_type: config::CrateType,
                      outputs: &OutputFilenames,
                      crate_name: &str) -> Vec<PathBuf> {
    for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
        check_file_is_writeable(obj, sess);
    }

    let mut out_filenames = vec![];

    if outputs.outputs.contains_key(&OutputType::Metadata) {
        let out_filename = filename_for_metadata(sess, crate_name, outputs);
        // To avoid races with another rustc process scanning the output directory,
        // we need to write the file somewhere else and atomically move it to its
        // final destination, with a `fs::rename` call. In order for the rename to
        // always succeed, the temporary file needs to be on the same filesystem,
        // which is why we create it inside the output directory specifically.
        let metadata_tmpdir = match TempFileBuilder::new()
            .prefix("rmeta")
            .tempdir_in(out_filename.parent().unwrap())
        {
            Ok(tmpdir) => tmpdir,
            Err(err) => sess.fatal(&format!("couldn't create a temp dir: {}", err)),
        };
        let metadata = emit_metadata(sess, codegen_results, &metadata_tmpdir);
        if let Err(e) = fs::rename(metadata, &out_filename) {
            sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
        }
        out_filenames.push(out_filename);
    }

    let tmpdir = match TempFileBuilder::new().prefix("rustc").tempdir() {
        Ok(tmpdir) => tmpdir,
        Err(err) => sess.fatal(&format!("couldn't create a temp dir: {}", err)),
    };

    if outputs.outputs.should_codegen() {
        let out_filename = out_filename(sess, crate_type, outputs, crate_name);
        match crate_type {
            config::CrateType::Rlib => {
                link_rlib(sess,
                          codegen_results,
                          RlibFlavor::Normal,
                          &out_filename,
                          &tmpdir).build();
            }
            config::CrateType::Staticlib => {
                link_staticlib(sess, codegen_results, &out_filename, &tmpdir);
            }
            _ => {
                link_natively(sess, crate_type, &out_filename, codegen_results, tmpdir.path());
            }
        }
        out_filenames.push(out_filename);
    }

    if sess.opts.cg.save_temps {
        let _ = tmpdir.into_path();
    }

    out_filenames
}

fn archive_search_paths(sess: &Session) -> Vec<PathBuf> {
    let mut search = Vec::new();
    sess.target_filesearch(PathKind::Native).for_each_lib_search_path(|path, _| {
        search.push(path.to_path_buf());
    });
    return search;
}

fn archive_config<'a>(sess: &'a Session,
                      output: &Path,
                      input: Option<&Path>) -> ArchiveConfig<'a> {
    ArchiveConfig {
        sess,
        dst: output.to_path_buf(),
        src: input.map(|p| p.to_path_buf()),
        lib_search_paths: archive_search_paths(sess),
    }
}

/// We use a temp directory here to avoid races between concurrent rustc processes,
/// such as builds in the same directory using the same filename for metadata while
/// building an `.rlib` (stomping over one another), or writing an `.rmeta` into a
/// directory being searched for `extern crate` (observing an incomplete file).
/// The returned path is the temporary file containing the complete metadata.
fn emit_metadata<'a>(sess: &'a Session, codegen_results: &CodegenResults, tmpdir: &TempDir)
                     -> PathBuf {
    let out_filename = tmpdir.path().join(METADATA_FILENAME);
    let result = fs::write(&out_filename, &codegen_results.metadata.raw_data);

    if let Err(e) = result {
        sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
    }

    out_filename
}

enum RlibFlavor {
    Normal,
    StaticlibBase,
}

// Create an 'rlib'
//
// An rlib in its current incarnation is essentially a renamed .a file. The
// rlib primarily contains the object file of the crate, but it also contains
// all of the object files from native libraries. This is done by unzipping
// native libraries and inserting all of the contents into this archive.
fn link_rlib<'a>(sess: &'a Session,
                 codegen_results: &CodegenResults,
                 flavor: RlibFlavor,
                 out_filename: &Path,
                 tmpdir: &TempDir) -> ArchiveBuilder<'a> {
    info!("preparing rlib to {:?}", out_filename);
    let mut ab = ArchiveBuilder::new(archive_config(sess, out_filename, None));

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
            NativeLibraryKind::NativeStatic => {}
            NativeLibraryKind::NativeStaticNobundle |
            NativeLibraryKind::NativeFramework |
            NativeLibraryKind::NativeUnknown => continue,
        }
        if let Some(name) = lib.name {
            ab.add_native_library(&name.as_str());
        }
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
            // Instead of putting the metadata in an object file section, rlibs
            // contain the metadata in a separate file.
            ab.add_file(&emit_metadata(sess, codegen_results, tmpdir));

            // For LTO purposes, the bytecode of this library is also inserted
            // into the archive.
            for bytecode in codegen_results
                .modules
                .iter()
                .filter_map(|m| m.bytecode_compressed.as_ref())
            {
                ab.add_file(bytecode);
            }

            // After adding all files to the archive, we need to update the
            // symbol table of the archive. This currently dies on macOS (see
            // #11162), and isn't necessary there anyway
            if !sess.target.target.options.is_like_osx {
                ab.update_symbols();
            }
        }

        RlibFlavor::StaticlibBase => {
            let obj = codegen_results.allocator_module
                .as_ref()
                .and_then(|m| m.object.as_ref());
            if let Some(obj) = obj {
                ab.add_file(obj);
            }
        }
    }

    ab
}

// Create a static archive
//
// This is essentially the same thing as an rlib, but it also involves adding
// all of the upstream crates' objects into the archive. This will slurp in
// all of the native libraries of upstream dependencies as well.
//
// Additionally, there's no way for us to link dynamic libraries, so we warn
// about all dynamic library dependencies that they're not linked in.
//
// There's no need to include metadata in a static archive, so ensure to not
// link in the metadata object file (and also don't prepare the archive with a
// metadata file).
fn link_staticlib(sess: &Session,
                  codegen_results: &CodegenResults,
                  out_filename: &Path,
                  tempdir: &TempDir) {
    let mut ab = link_rlib(sess,
                           codegen_results,
                           RlibFlavor::StaticlibBase,
                           out_filename,
                           tempdir);
    let mut all_native_libs = vec![];

    let res = each_linked_rlib(sess, &codegen_results.crate_info, &mut |cnum, path| {
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
            lib.kind == NativeLibraryKind::NativeStatic && !relevant_lib(sess, lib)
        });
        ab.add_rlib(path,
                    &name.as_str(),
                    are_upstream_rust_objects_already_included(sess) &&
                        !ignored_for_lto(sess, &codegen_results.crate_info, cnum),
                    skip_object_files).unwrap();

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
}

fn print_native_static_libs(sess: &Session, all_native_libs: &[NativeLibrary]) {
    let lib_args: Vec<_> = all_native_libs.iter()
        .filter(|l| relevant_lib(sess, l))
        .filter_map(|lib| {
            let name = lib.name?;
            match lib.kind {
                NativeLibraryKind::NativeStaticNobundle |
                NativeLibraryKind::NativeUnknown => {
                    if sess.target.target.options.is_like_msvc {
                        Some(format!("{}.lib", name))
                    } else {
                        Some(format!("-l{}", name))
                    }
                },
                NativeLibraryKind::NativeFramework => {
                    // ld-only syntax, since there are no frameworks in MSVC
                    Some(format!("-framework {}", name))
                },
                // These are included, no need to print them
                NativeLibraryKind::NativeStatic => None,
            }
        })
        .collect();
    if !lib_args.is_empty() {
        sess.note_without_error("Link against the following native artifacts when linking \
                                 against this static library. The order and any duplication \
                                 can be significant on some platforms.");
        // Prefix for greppability
        sess.note_without_error(&format!("native-static-libs: {}", &lib_args.join(" ")));
    }
}

pub fn linker_and_flavor(sess: &Session) -> (PathBuf, LinkerFlavor) {
    fn infer_from(
        sess: &Session,
        linker: Option<PathBuf>,
        flavor: Option<LinkerFlavor>,
    ) -> Option<(PathBuf, LinkerFlavor)> {
        match (linker, flavor) {
            (Some(linker), Some(flavor)) => Some((linker, flavor)),
            // only the linker flavor is known; use the default linker for the selected flavor
            (None, Some(flavor)) => Some((PathBuf::from(match flavor {
                LinkerFlavor::Em  => if cfg!(windows) { "emcc.bat" } else { "emcc" },
                LinkerFlavor::Gcc => "cc",
                LinkerFlavor::Ld => "ld",
                LinkerFlavor::Msvc => "link.exe",
                LinkerFlavor::Lld(_) => "lld",
            }), flavor)),
            (Some(linker), None) => {
                let stem = linker.file_stem().and_then(|stem| stem.to_str()).unwrap_or_else(|| {
                    sess.fatal("couldn't extract file stem from specified linker");
                }).to_owned();

                let flavor = if stem == "emcc" {
                    LinkerFlavor::Em
                } else if stem == "gcc" || stem.ends_with("-gcc") {
                    LinkerFlavor::Gcc
                } else if stem == "ld" || stem == "ld.lld" || stem.ends_with("-ld") {
                    LinkerFlavor::Ld
                } else if stem == "link" || stem == "lld-link" {
                    LinkerFlavor::Msvc
                } else if stem == "lld" || stem == "rust-lld" {
                    LinkerFlavor::Lld(sess.target.target.options.lld_flavor)
                } else {
                    // fall back to the value in the target spec
                    sess.target.target.linker_flavor
                };

                Some((linker, flavor))
            },
            (None, None) => None,
        }
    }

    // linker and linker flavor specified via command line have precedence over what the target
    // specification specifies
    if let Some(ret) = infer_from(
        sess,
        sess.opts.cg.linker.clone(),
        sess.opts.debugging_opts.linker_flavor,
    ) {
        return ret;
    }

    if let Some(ret) = infer_from(
        sess,
        sess.target.target.options.linker.clone().map(PathBuf::from),
        Some(sess.target.target.linker_flavor),
    ) {
        return ret;
    }

    bug!("Not enough information provided to determine how to invoke the linker");
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively(sess: &Session,
                 crate_type: config::CrateType,
                 out_filename: &Path,
                 codegen_results: &CodegenResults,
                 tmpdir: &Path) {
    info!("preparing {:?} to {:?}", crate_type, out_filename);
    let (linker, flavor) = linker_and_flavor(sess);

    // The invocations of cc share some flags across platforms
    let (pname, mut cmd) = get_linker(sess, &linker, flavor);

    let root = sess.target_filesearch(PathKind::Native).get_lib_path();
    if let Some(args) = sess.target.target.options.pre_link_args.get(&flavor) {
        cmd.args(args);
    }
    if let Some(args) = sess.target.target.options.pre_link_args_crt.get(&flavor) {
        if sess.crt_static() {
            cmd.args(args);
        }
    }
    if let Some(ref args) = sess.opts.debugging_opts.pre_link_args {
        cmd.args(args);
    }
    cmd.args(&sess.opts.debugging_opts.pre_link_arg);

    let pre_link_objects = if crate_type == config::CrateType::Executable {
        &sess.target.target.options.pre_link_objects_exe
    } else {
        &sess.target.target.options.pre_link_objects_dll
    };
    for obj in pre_link_objects {
        cmd.arg(root.join(obj));
    }

    if crate_type == config::CrateType::Executable && sess.crt_static() {
        for obj in &sess.target.target.options.pre_link_objects_exe_crt {
            cmd.arg(root.join(obj));
        }
    }

    if sess.target.target.options.is_like_emscripten {
        cmd.arg("-s");
        cmd.arg(if sess.panic_strategy() == PanicStrategy::Abort {
            "DISABLE_EXCEPTION_CATCHING=1"
        } else {
            "DISABLE_EXCEPTION_CATCHING=0"
        });
    }

    {
        let mut linker = codegen_results.linker_info.to_linker(cmd, &sess, flavor);
        link_args(&mut *linker, flavor, sess, crate_type, tmpdir,
                  out_filename, codegen_results);
        cmd = linker.finalize();
    }
    if let Some(args) = sess.target.target.options.late_link_args.get(&flavor) {
        cmd.args(args);
    }
    for obj in &sess.target.target.options.post_link_objects {
        cmd.arg(root.join(obj));
    }
    if sess.crt_static() {
        for obj in &sess.target.target.options.post_link_objects_crt {
            cmd.arg(root.join(obj));
        }
    }
    if let Some(args) = sess.target.target.options.post_link_args.get(&flavor) {
        cmd.args(args);
    }
    for &(ref k, ref v) in &sess.target.target.options.link_env {
        cmd.env(k, v);
    }

    if sess.opts.debugging_opts.print_link_args {
        println!("{:?}", &cmd);
    }

    // May have not found libraries in the right formats.
    sess.abort_if_errors();

    // Invoke the system linker
    //
    // Note that there's a terribly awful hack that really shouldn't be present
    // in any compiler. Here an environment variable is supported to
    // automatically retry the linker invocation if the linker looks like it
    // segfaulted.
    //
    // Gee that seems odd, normally segfaults are things we want to know about!
    // Unfortunately though in rust-lang/rust#38878 we're experiencing the
    // linker segfaulting on Travis quite a bit which is causing quite a bit of
    // pain to land PRs when they spuriously fail due to a segfault.
    //
    // The issue #38878 has some more debugging information on it as well, but
    // this unfortunately looks like it's just a race condition in macOS's linker
    // with some thread pool working in the background. It seems that no one
    // currently knows a fix for this so in the meantime we're left with this...
    info!("{:?}", &cmd);
    let retry_on_segfault = env::var("RUSTC_RETRY_LINKER_ON_SEGFAULT").is_ok();
    let mut prog;
    let mut i = 0;
    loop {
        i += 1;
        prog = time(sess, "running linker", || {
            exec_linker(sess, &mut cmd, out_filename, tmpdir)
        });
        let output = match prog {
            Ok(ref output) => output,
            Err(_) => break,
        };
        if output.status.success() {
            break
        }
        let mut out = output.stderr.clone();
        out.extend(&output.stdout);
        let out = String::from_utf8_lossy(&out);

        // Check to see if the link failed with "unrecognized command line option:
        // '-no-pie'" for gcc or "unknown argument: '-no-pie'" for clang. If so,
        // reperform the link step without the -no-pie option. This is safe because
        // if the linker doesn't support -no-pie then it should not default to
        // linking executables as pie. Different versions of gcc seem to use
        // different quotes in the error message so don't check for them.
        if sess.target.target.options.linker_is_gnu &&
           flavor != LinkerFlavor::Ld &&
           (out.contains("unrecognized command line option") ||
            out.contains("unknown argument")) &&
           out.contains("-no-pie") &&
           cmd.get_args().iter().any(|e| e.to_string_lossy() == "-no-pie") {
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
        if !retry_on_segfault || i > 3 {
            break
        }
        let msg_segv = "clang: error: unable to execute command: Segmentation fault: 11";
        let msg_bus  = "clang: error: unable to execute command: Bus error: 10";
        if !(out.contains(msg_segv) || out.contains(msg_bus)) {
            break
        }

        warn!(
            "looks like the linker segfaulted when we tried to call it, \
             automatically retrying again. cmd = {:?}, out = {}.",
            cmd,
            out,
        );
    }

    match prog {
        Ok(prog) => {
            fn escape_string(s: &[u8]) -> String {
                str::from_utf8(s).map(|s| s.to_owned())
                    .unwrap_or_else(|_| {
                        let mut x = "Non-UTF-8 output: ".to_string();
                        x.extend(s.iter()
                                 .flat_map(|&b| ascii::escape_default(b))
                                 .map(|b| char::from_u32(b as u32).unwrap()));
                        x
                    })
            }
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                sess.struct_err(&format!("linking with `{}` failed: {}",
                                         pname.display(),
                                         prog.status))
                    .note(&format!("{:?}", &cmd))
                    .note(&escape_string(&output))
                    .emit();
                sess.abort_if_errors();
            }
            info!("linker stderr:\n{}", escape_string(&prog.stderr));
            info!("linker stdout:\n{}", escape_string(&prog.stdout));
        },
        Err(e) => {
            let linker_not_found = e.kind() == io::ErrorKind::NotFound;

            let mut linker_error = {
                if linker_not_found {
                    sess.struct_err(&format!("linker `{}` not found", pname.display()))
                } else {
                    sess.struct_err(&format!("could not exec the linker `{}`", pname.display()))
                }
            };

            linker_error.note(&e.to_string());

            if !linker_not_found {
                linker_error.note(&format!("{:?}", &cmd));
            }

            linker_error.emit();

            if sess.target.target.options.is_like_msvc && linker_not_found {
                sess.note_without_error("the msvc targets depend on the msvc linker \
                    but `link.exe` was not found");
                sess.note_without_error("please ensure that VS 2013, VS 2015 or VS 2017 \
                    was installed with the Visual C++ option");
            }
            sess.abort_if_errors();
        }
    }


    // On macOS, debuggers need this utility to get run to do some munging of
    // the symbols. Note, though, that if the object files are being preserved
    // for their debug information there's no need for us to run dsymutil.
    if sess.target.target.options.is_like_osx &&
        sess.opts.debuginfo != DebugInfo::None &&
        !preserve_objects_for_their_debuginfo(sess)
    {
        match Command::new("dsymutil").arg(out_filename).output() {
            Ok(..) => {}
            Err(e) => sess.fatal(&format!("failed to run dsymutil: {}", e)),
        }
    }

    if sess.opts.target_triple.triple() == "wasm32-unknown-unknown" {
        wasm::rewrite_imports(&out_filename, &codegen_results.crate_info.wasm_imports);
    }
}

fn exec_linker(sess: &Session, cmd: &mut Command, out_filename: &Path, tmpdir: &Path)
    -> io::Result<Output>
{
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
            Err(e) => return Err(e)
        }
    }

    info!("falling back to passing arguments to linker via an @-file");
    let mut cmd2 = cmd.clone();
    let mut args = String::new();
    for arg in cmd2.take_args() {
        args.push_str(&Escape {
            arg: arg.to_str().unwrap(),
            is_like_msvc: sess.target.target.options.is_like_msvc,
        }.to_string());
        args.push_str("\n");
    }
    let file = tmpdir.join("linker-arguments");
    let bytes = if sess.target.target.options.is_like_msvc {
        let mut out = Vec::with_capacity((1 + args.len()) * 2);
        // start the stream with a UTF-16 BOM
        for c in iter::once(0xFEFF).chain(args.encode_utf16()) {
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

    #[cfg(unix)]
    fn flush_linked_file(_: &io::Result<Output>, _: &Path) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn flush_linked_file(command_output: &io::Result<Output>, out_filename: &Path)
        -> io::Result<()>
    {
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

    struct Escape<'a> {
        arg: &'a str,
        is_like_msvc: bool,
    }

    impl<'a> fmt::Display for Escape<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.is_like_msvc {
                // This is "documented" at
                // https://msdn.microsoft.com/en-us/library/4xdcbak7.aspx
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
                        '\\' |
                        ' ' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
            }
            Ok(())
        }
    }
}

fn link_args(cmd: &mut dyn Linker,
             flavor: LinkerFlavor,
             sess: &Session,
             crate_type: config::CrateType,
             tmpdir: &Path,
             out_filename: &Path,
             codegen_results: &CodegenResults) {

    // Linker plugins should be specified early in the list of arguments
    cmd.cross_lang_lto();

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = sess.target_filesearch(PathKind::All).get_lib_path();

    // target descriptor
    let t = &sess.target.target;

    cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));
    for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
        cmd.add_object(obj);
    }
    cmd.output_filename(out_filename);

    if crate_type == config::CrateType::Executable &&
       sess.target.target.options.is_like_windows {
        if let Some(ref s) = codegen_results.windows_subsystem {
            cmd.subsystem(s);
        }
    }

    // If we're building a dynamic library then some platforms need to make sure
    // that all symbols are exported correctly from the dynamic library.
    if crate_type != config::CrateType::Executable ||
       sess.target.target.options.is_like_emscripten {
        cmd.export_symbols(tmpdir, crate_type);
    }

    // When linking a dynamic library, we put the metadata into a section of the
    // executable. This metadata is in a separate object file from the main
    // object file, so we link that in here.
    if crate_type == config::CrateType::Dylib ||
       crate_type == config::CrateType::ProcMacro {
        if let Some(obj) = codegen_results.metadata_module.object.as_ref() {
            cmd.add_object(obj);
        }
    }

    let obj = codegen_results.allocator_module
        .as_ref()
        .and_then(|m| m.object.as_ref());
    if let Some(obj) = obj {
        cmd.add_object(obj);
    }

    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    if !sess.opts.cg.link_dead_code {
        let keep_metadata = crate_type == config::CrateType::Dylib;
        cmd.gc_sections(keep_metadata);
    }

    let used_link_args = &codegen_results.crate_info.link_args;

    if crate_type == config::CrateType::Executable {
        let mut position_independent_executable = false;

        if t.options.position_independent_executables {
            let empty_vec = Vec::new();
            let args = sess.opts.cg.link_args.as_ref().unwrap_or(&empty_vec);
            let more_args = &sess.opts.cg.link_arg;
            let mut args = args.iter().chain(more_args.iter()).chain(used_link_args.iter());

            if get_reloc_model(sess) == llvm::RelocMode::PIC
                && !sess.crt_static() && !args.any(|x| *x == "-static") {
                position_independent_executable = true;
            }
        }

        if position_independent_executable {
            cmd.position_independent_executable();
        } else {
            // recent versions of gcc can be configured to generate position
            // independent executables by default. We have to pass -no-pie to
            // explicitly turn that off. Not applicable to ld.
            if sess.target.target.options.linker_is_gnu
                && flavor != LinkerFlavor::Ld {
                cmd.no_position_independent_executable();
            }
        }
    }

    let relro_level = match sess.opts.debugging_opts.relro_level {
        Some(level) => level,
        None => t.options.relro_level,
    };
    match relro_level {
        RelroLevel::Full => {
            cmd.full_relro();
        },
        RelroLevel::Partial => {
            cmd.partial_relro();
        },
        RelroLevel::Off => {
            cmd.no_relro();
        },
        RelroLevel::None => {
        },
    }

    // Pass optimization flags down to the linker.
    cmd.optimize();

    // Pass debuginfo flags down to the linker.
    cmd.debuginfo();

    // We want to, by default, prevent the compiler from accidentally leaking in
    // any system libraries, so we may explicitly ask linkers to not link to any
    // libraries by default. Note that this does not happen for windows because
    // windows pulls in some large number of libraries and I couldn't quite
    // figure out which subset we wanted.
    //
    // This is all naturally configurable via the standard methods as well.
    if !sess.opts.cg.default_linker_libraries.unwrap_or(false) &&
        t.options.no_default_libraries
    {
        cmd.no_default_libraries();
    }

    // Take careful note of the ordering of the arguments we pass to the linker
    // here. Linkers will assume that things on the left depend on things to the
    // right. Things on the right cannot depend on things on the left. This is
    // all formally implemented in terms of resolving symbols (libs on the right
    // resolve unknown symbols of libs on the left, but not vice versa).
    //
    // For this reason, we have organized the arguments we pass to the linker as
    // such:
    //
    //  1. The local object that LLVM just generated
    //  2. Local native libraries
    //  3. Upstream rust libraries
    //  4. Upstream native libraries
    //
    // The rationale behind this ordering is that those items lower down in the
    // list can't depend on items higher up in the list. For example nothing can
    // depend on what we just generated (e.g. that'd be a circular dependency).
    // Upstream rust libraries are not allowed to depend on our local native
    // libraries as that would violate the structure of the DAG, in that
    // scenario they are required to link to them as well in a shared fashion.
    //
    // Note that upstream rust libraries may contain native dependencies as
    // well, but they also can't depend on what we just started to add to the
    // link line. And finally upstream native libraries can't depend on anything
    // in this DAG so far because they're only dylibs and dylibs can only depend
    // on other dylibs (e.g. other native deps).
    add_local_native_libraries(cmd, sess, codegen_results);
    add_upstream_rust_crates(cmd, sess, codegen_results, crate_type, tmpdir);
    add_upstream_native_libraries(cmd, sess, codegen_results, crate_type);

    // Tell the linker what we're doing.
    if crate_type != config::CrateType::Executable {
        cmd.build_dylib(out_filename);
    }
    if crate_type == config::CrateType::Executable && sess.crt_static() {
        cmd.build_static_executable();
    }

    if sess.opts.debugging_opts.pgo_gen.is_some() {
        cmd.pgo_gen();
    }

    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // addl_lib_search_paths
    if sess.opts.cg.rpath {
        let sysroot = sess.sysroot();
        let target_triple = sess.opts.target_triple.triple();
        let mut get_install_prefix_lib_path = || {
            let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");
            let tlib = filesearch::relative_target_lib_path(sysroot, target_triple);
            let mut path = PathBuf::from(install_prefix);
            path.push(&tlib);

            path
        };
        let mut rpath_config = RPathConfig {
            used_crates: &codegen_results.crate_info.used_crates_dynamic,
            out_filename: out_filename.to_path_buf(),
            has_rpath: sess.target.target.options.has_rpath,
            is_like_osx: sess.target.target.options.is_like_osx,
            linker_is_gnu: sess.target.target.options.linker_is_gnu,
            get_install_prefix_lib_path: &mut get_install_prefix_lib_path,
        };
        cmd.args(&rpath::get_rpath_flags(&mut rpath_config));
    }

    // Finally add all the linker arguments provided on the command line along
    // with any #[link_args] attributes found inside the crate
    if let Some(ref args) = sess.opts.cg.link_args {
        cmd.args(args);
    }
    cmd.args(&sess.opts.cg.link_arg);
    cmd.args(&used_link_args);
}

// # Native library linking
//
// User-supplied library search paths (-L on the command line). These are
// the same paths used to find Rust crates, so some of them may have been
// added already by the previous crate linking code. This only allows them
// to be found at compile time so it is still entirely up to outside
// forces to make sure that library can be found at runtime.
//
// Also note that the native libraries linked here are only the ones located
// in the current crate. Upstream crates with native library dependencies
// may have their native library pulled in above.
fn add_local_native_libraries(cmd: &mut dyn Linker,
                              sess: &Session,
                              codegen_results: &CodegenResults) {
    sess.target_filesearch(PathKind::All).for_each_lib_search_path(|path, k| {
        match k {
            PathKind::Framework => { cmd.framework_path(path); }
            _ => { cmd.include_path(&fix_windows_verbatim_for_gcc(path)); }
        }
    });

    let relevant_libs = codegen_results.crate_info.used_libraries.iter().filter(|l| {
        relevant_lib(sess, l)
    });

    let search_path = archive_search_paths(sess);
    for lib in relevant_libs {
        let name = match lib.name {
            Some(ref l) => l,
            None => continue,
        };
        match lib.kind {
            NativeLibraryKind::NativeUnknown => cmd.link_dylib(&name.as_str()),
            NativeLibraryKind::NativeFramework => cmd.link_framework(&name.as_str()),
            NativeLibraryKind::NativeStaticNobundle => cmd.link_staticlib(&name.as_str()),
            NativeLibraryKind::NativeStatic => cmd.link_whole_staticlib(&name.as_str(),
                                                                        &search_path)
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
fn add_upstream_rust_crates(cmd: &mut dyn Linker,
                            sess: &Session,
                            codegen_results: &CodegenResults,
                            crate_type: config::CrateType,
                            tmpdir: &Path) {
    // All of the heavy lifting has previously been accomplished by the
    // dependency_format module of the compiler. This is just crawling the
    // output of that module, adding crates as necessary.
    //
    // Linking to a rlib involves just passing it to the linker (the linker
    // will slurp up the object files inside), and linking to a dynamic library
    // involves just passing the right -l flag.

    let formats = sess.dependency_formats.borrow();
    let data = formats.get(&crate_type).unwrap();

    // Invoke get_used_crates to ensure that we get a topological sorting of
    // crates.
    let deps = &codegen_results.crate_info.used_crates_dynamic;

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
    let mut end_with = FxHashSet();
    let info = &codegen_results.crate_info;
    for &(cnum, _) in deps.iter().rev() {
        if let Some(missing) = info.missing_lang_items.get(&cnum) {
            end_with.extend(missing.iter().cloned());
            if end_with.len() > 0 && group_end.is_none() {
                group_end = Some(cnum);
            }
        }
        end_with.retain(|item| info.lang_item_to_crate.get(item) != Some(&cnum));
        if end_with.len() == 0 && group_end.is_some() {
            group_start = Some(cnum);
            break
        }
    }

    // If we didn't end up filling in all lang items from upstream crates then
    // we'll be filling it in with our crate. This probably means we're the
    // standard library itself, so skip this for now.
    if group_end.is_some() && group_start.is_none() {
        group_end = None;
    }

    let mut compiler_builtins = None;

    for &(cnum, _) in deps.iter() {
        if group_start == Some(cnum) {
            cmd.group_start();
        }

        // We may not pass all crates through to the linker. Some crates may
        // appear statically in an existing dylib, meaning we'll pick up all the
        // symbols from the dylib.
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        match data[cnum.as_usize() - 1] {
            _ if codegen_results.crate_info.profiler_runtime == Some(cnum) => {
                add_static_crate(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
            }
            _ if codegen_results.crate_info.sanitizer_runtime == Some(cnum) => {
                link_sanitizer_runtime(cmd, sess, codegen_results, tmpdir, cnum);
            }
            // compiler-builtins are always placed last to ensure that they're
            // linked correctly.
            _ if codegen_results.crate_info.compiler_builtins == Some(cnum) => {
                assert!(compiler_builtins.is_none());
                compiler_builtins = Some(cnum);
            }
            Linkage::NotLinked |
            Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                add_static_crate(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
            }
            Linkage::Dynamic => {
                add_dynamic_crate(cmd, sess, &src.dylib.as_ref().unwrap().0)
            }
        }

        if group_end == Some(cnum) {
            cmd.group_end();
        }
    }

    // compiler-builtins are always placed last to ensure that they're
    // linked correctly.
    // We must always link the `compiler_builtins` crate statically. Even if it
    // was already "included" in a dylib (e.g. `libstd` when `-C prefer-dynamic`
    // is used)
    if let Some(cnum) = compiler_builtins {
        add_static_crate(cmd, sess, codegen_results, tmpdir, crate_type, cnum);
    }

    // Converts a library file-stem into a cc -l argument
    fn unlib<'a>(config: &config::Config, stem: &'a str) -> &'a str {
        if stem.starts_with("lib") && !config.target.options.is_like_windows {
            &stem[3..]
        } else {
            stem
        }
    }

    // We must link the sanitizer runtime using -Wl,--whole-archive but since
    // it's packed in a .rlib, it contains stuff that are not objects that will
    // make the linker error. So we must remove those bits from the .rlib before
    // linking it.
    fn link_sanitizer_runtime(cmd: &mut dyn Linker,
                              sess: &Session,
                              codegen_results: &CodegenResults,
                              tmpdir: &Path,
                              cnum: CrateNum) {
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        let cratepath = &src.rlib.as_ref().unwrap().0;

        if sess.target.target.options.is_like_osx {
            // On Apple platforms, the sanitizer is always built as a dylib, and
            // LLVM will link to `@rpath/*.dylib`, so we need to specify an
            // rpath to the library as well (the rpath should be absolute, see
            // PR #41352 for details).
            //
            // FIXME: Remove this logic into librustc_*san once Cargo supports it
            let rpath = cratepath.parent().unwrap();
            let rpath = rpath.to_str().expect("non-utf8 component in path");
            cmd.args(&["-Wl,-rpath".into(), "-Xlinker".into(), rpath.into()]);
        }

        let dst = tmpdir.join(cratepath.file_name().unwrap());
        let cfg = archive_config(sess, &dst, Some(cratepath));
        let mut archive = ArchiveBuilder::new(cfg);
        archive.update_symbols();

        for f in archive.src_files() {
            if f.ends_with(RLIB_BYTECODE_EXTENSION) || f == METADATA_FILENAME {
                archive.remove_file(&f);
                continue
            }
        }

        archive.build();

        cmd.link_whole_rlib(&dst);
    }

    // Adds the static "rlib" versions of all crates to the command line.
    // There's a bit of magic which happens here specifically related to LTO and
    // dynamic libraries. Specifically:
    //
    // * For LTO, we remove upstream object files.
    // * For dylibs we remove metadata and bytecode from upstream rlibs
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
    // When making a dynamic library, linkers by default don't include any
    // object files in an archive if they're not necessary to resolve the link.
    // We basically want to convert the archive (rlib) to a dylib, though, so we
    // *do* want everything included in the output, regardless of whether the
    // linker thinks it's needed or not. As a result we must use the
    // --whole-archive option (or the platform equivalent). When using this
    // option the linker will fail if there are non-objects in the archive (such
    // as our own metadata and/or bytecode). All in all, for rlibs to be
    // entirely included in dylibs, we need to remove all non-object files.
    //
    // Note, however, that if we're not doing LTO or we're not producing a dylib
    // (aka we're making an executable), we can just pass the rlib blindly to
    // the linker (fast) because it's fine if it's not actually included as
    // we're at the end of the dependency chain.
    fn add_static_crate(cmd: &mut dyn Linker,
                        sess: &Session,
                        codegen_results: &CodegenResults,
                        tmpdir: &Path,
                        crate_type: config::CrateType,
                        cnum: CrateNum) {
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        let cratepath = &src.rlib.as_ref().unwrap().0;

        // See the comment above in `link_staticlib` and `link_rlib` for why if
        // there's a static library that's not relevant we skip all object
        // files.
        let native_libs = &codegen_results.crate_info.native_libraries[&cnum];
        let skip_native = native_libs.iter().any(|lib| {
            lib.kind == NativeLibraryKind::NativeStatic && !relevant_lib(sess, lib)
        });

        if (!are_upstream_rust_objects_already_included(sess) ||
            ignored_for_lto(sess, &codegen_results.crate_info, cnum)) &&
           crate_type != config::CrateType::Dylib &&
           !skip_native {
            cmd.link_rlib(&fix_windows_verbatim_for_gcc(cratepath));
            return
        }

        let dst = tmpdir.join(cratepath.file_name().unwrap());
        let name = cratepath.file_name().unwrap().to_str().unwrap();
        let name = &name[3..name.len() - 5]; // chop off lib/.rlib

        time(sess, &format!("altering {}.rlib", name), || {
            let cfg = archive_config(sess, &dst, Some(cratepath));
            let mut archive = ArchiveBuilder::new(cfg);
            archive.update_symbols();

            let mut any_objects = false;
            for f in archive.src_files() {
                if f.ends_with(RLIB_BYTECODE_EXTENSION) || f == METADATA_FILENAME {
                    archive.remove_file(&f);
                    continue
                }

                let canonical = f.replace("-", "_");
                let canonical_name = name.replace("-", "_");

                // Look for `.rcgu.o` at the end of the filename to conclude
                // that this is a Rust-related object file.
                fn looks_like_rust(s: &str) -> bool {
                    let path = Path::new(s);
                    let ext = path.extension().and_then(|s| s.to_str());
                    if ext != Some(OutputType::Object.extension()) {
                        return false
                    }
                    let ext2 = path.file_stem()
                        .and_then(|s| Path::new(s).extension())
                        .and_then(|s| s.to_str());
                    ext2 == Some(RUST_CGU_EXT)
                }

                let is_rust_object =
                    canonical.starts_with(&canonical_name) &&
                    looks_like_rust(&f);

                // If we've been requested to skip all native object files
                // (those not generated by the rust compiler) then we can skip
                // this file. See above for why we may want to do this.
                let skip_because_cfg_say_so = skip_native && !is_rust_object;

                // If we're performing LTO and this is a rust-generated object
                // file, then we don't need the object file as it's part of the
                // LTO module. Note that `#![no_builtins]` is excluded from LTO,
                // though, so we let that object file slide.
                let skip_because_lto = are_upstream_rust_objects_already_included(sess) &&
                    is_rust_object &&
                    (sess.target.target.options.no_builtins ||
                     !codegen_results.crate_info.is_no_builtins.contains(&cnum));

                if skip_because_cfg_say_so || skip_because_lto {
                    archive.remove_file(&f);
                } else {
                    any_objects = true;
                }
            }

            if !any_objects {
                return
            }
            archive.build();

            // If we're creating a dylib, then we need to include the
            // whole of each object in our archive into that artifact. This is
            // because a `dylib` can be reused as an intermediate artifact.
            //
            // Note, though, that we don't want to include the whole of a
            // compiler-builtins crate (e.g. compiler-rt) because it'll get
            // repeatedly linked anyway.
            if crate_type == config::CrateType::Dylib &&
                codegen_results.crate_info.compiler_builtins != Some(cnum) {
                cmd.link_whole_rlib(&fix_windows_verbatim_for_gcc(&dst));
            } else {
                cmd.link_rlib(&fix_windows_verbatim_for_gcc(&dst));
            }
        });
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crate(cmd: &mut dyn Linker, sess: &Session, cratepath: &Path) {
        // If we're performing LTO, then it should have been previously required
        // that all upstream rust dependencies were available in an rlib format.
        assert!(!are_upstream_rust_objects_already_included(sess));

        // Just need to tell the linker about where the library lives and
        // what its name is
        let parent = cratepath.parent();
        if let Some(dir) = parent {
            cmd.include_path(&fix_windows_verbatim_for_gcc(dir));
        }
        let filestem = cratepath.file_stem().unwrap().to_str().unwrap();
        cmd.link_rust_dylib(&unlib(&sess.target, filestem),
                            parent.unwrap_or(Path::new("")));
    }
}

// Link in all of our upstream crates' native dependencies. Remember that
// all of these upstream native dependencies are all non-static
// dependencies. We've got two cases then:
//
// 1. The upstream crate is an rlib. In this case we *must* link in the
// native dependency because the rlib is just an archive.
//
// 2. The upstream crate is a dylib. In order to use the dylib, we have to
// have the dependency present on the system somewhere. Thus, we don't
// gain a whole lot from not linking in the dynamic dependency to this
// crate as well.
//
// The use case for this is a little subtle. In theory the native
// dependencies of a crate are purely an implementation detail of the crate
// itself, but the problem arises with generic and inlined functions. If a
// generic function calls a native function, then the generic function must
// be instantiated in the target crate, meaning that the native symbol must
// also be resolved in the target crate.
fn add_upstream_native_libraries(cmd: &mut dyn Linker,
                                 sess: &Session,
                                 codegen_results: &CodegenResults,
                                 crate_type: config::CrateType) {
    // Be sure to use a topological sorting of crates because there may be
    // interdependencies between native libraries. When passing -nodefaultlibs,
    // for example, almost all native libraries depend on libc, so we have to
    // make sure that's all the way at the right (liblibc is near the base of
    // the dependency chain).
    //
    // This passes RequireStatic, but the actual requirement doesn't matter,
    // we're just getting an ordering of crate numbers, we're not worried about
    // the paths.
    let formats = sess.dependency_formats.borrow();
    let data = formats.get(&crate_type).unwrap();

    let crates = &codegen_results.crate_info.used_crates_static;
    for &(cnum, _) in crates {
        for lib in codegen_results.crate_info.native_libraries[&cnum].iter() {
            let name = match lib.name {
                Some(ref l) => l,
                None => continue,
            };
            if !relevant_lib(sess, &lib) {
                continue
            }
            match lib.kind {
                NativeLibraryKind::NativeUnknown => cmd.link_dylib(&name.as_str()),
                NativeLibraryKind::NativeFramework => cmd.link_framework(&name.as_str()),
                NativeLibraryKind::NativeStaticNobundle => {
                    // Link "static-nobundle" native libs only if the crate they originate from
                    // is being linked statically to the current crate.  If it's linked dynamically
                    // or is an rlib already included via some other dylib crate, the symbols from
                    // native libs will have already been included in that dylib.
                    if data[cnum.as_usize() - 1] == Linkage::Static {
                        cmd.link_staticlib(&name.as_str())
                    }
                },
                // ignore statically included native libraries here as we've
                // already included them when we included the rust library
                // previously
                NativeLibraryKind::NativeStatic => {}
            }
        }
    }
}

fn relevant_lib(sess: &Session, lib: &NativeLibrary) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

fn are_upstream_rust_objects_already_included(sess: &Session) -> bool {
    match sess.lto() {
        Lto::Fat => true,
        Lto::Thin => {
            // If we defer LTO to the linker, we haven't run LTO ourselves, so
            // any upstream object files have not been copied yet.
            !sess.opts.debugging_opts.cross_lang_lto.enabled()
        }
        Lto::No |
        Lto::ThinLocal => false,
    }
}
