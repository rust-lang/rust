// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rustc_trans_utils;

use super::archive::{ArchiveBuilder, ArchiveConfig};
use super::linker::Linker;
use super::rpath::RPathConfig;
use super::rpath;
use metadata::METADATA_FILENAME;
use rustc::session::config::{self, NoDebugInfo, OutputFilenames, OutputType, PrintRequest};
use rustc::session::filesearch;
use rustc::session::search_paths::PathKind;
use rustc::session::Session;
use rustc::middle::cstore::{LinkMeta, NativeLibrary, LibSource, LinkagePreference,
                            NativeLibraryKind};
use rustc::middle::dependency_format::Linkage;
use CrateTranslation;
use rustc::util::common::time;
use rustc::util::fs::fix_windows_verbatim_for_gcc;
use rustc::dep_graph::{DepKind, DepNode};
use rustc::hir::def_id::CrateNum;
use rustc::hir::svh::Svh;
use rustc_back::tempdir::TempDir;
use rustc_back::{PanicStrategy, RelroLevel};
use rustc_incremental::IncrementalHashesMap;
use context::get_reloc_model;
use llvm;

use std::ascii;
use std::char;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use syntax::attr;

/// The LLVM module name containing crate-metadata. This includes a `.` on
/// purpose, so it cannot clash with the name of a user-defined module.
pub const METADATA_MODULE_NAME: &'static str = "crate.metadata";
/// The name of the crate-metadata object file the compiler generates. Must
/// match up with `METADATA_MODULE_NAME`.
pub const METADATA_OBJ_NAME: &'static str = "crate.metadata.o";

// same as for metadata above, but for allocator shim
pub const ALLOCATOR_MODULE_NAME: &'static str = "crate.allocator";
pub const ALLOCATOR_OBJ_NAME: &'static str = "crate.allocator.o";

// RLIB LLVM-BYTECODE OBJECT LAYOUT
// Version 1
// Bytes    Data
// 0..10    "RUST_OBJECT" encoded in ASCII
// 11..14   format version as little-endian u32
// 15..22   size in bytes of deflate compressed LLVM bitcode as
//          little-endian u64
// 23..     compressed LLVM bitcode

// This is the "magic number" expected at the beginning of a LLVM bytecode
// object in an rlib.
pub const RLIB_BYTECODE_OBJECT_MAGIC: &'static [u8] = b"RUST_OBJECT";

// The version number this compiler will write to bytecode objects in rlibs
pub const RLIB_BYTECODE_OBJECT_VERSION: u32 = 1;

// The offset in bytes the bytecode object format version number can be found at
pub const RLIB_BYTECODE_OBJECT_VERSION_OFFSET: usize = 11;

// The offset in bytes the size of the compressed bytecode can be found at in
// format version 1
pub const RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET: usize =
    RLIB_BYTECODE_OBJECT_VERSION_OFFSET + 4;

// The offset in bytes the compressed LLVM bytecode can be found at in format
// version 1
pub const RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET: usize =
    RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET + 8;

pub use self::rustc_trans_utils::link::{find_crate_name, filename_for_input,
                                        default_output_for_target, invalid_output_for_target};

pub fn build_link_meta(incremental_hashes_map: &IncrementalHashesMap) -> LinkMeta {
    let krate_dep_node = &DepNode::new_no_params(DepKind::Krate);
    let r = LinkMeta {
        crate_hash: Svh::new(incremental_hashes_map[krate_dep_node].to_smaller_hash()),
    };
    info!("{:?}", r);
    return r;
}

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
pub fn get_linker(sess: &Session) -> (String, Command, Vec<(OsString, OsString)>) {
    let envs = vec![("PATH".into(), command_path(sess))];

    if let Some(ref linker) = sess.opts.cg.linker {
        (linker.clone(), Command::new(linker), envs)
    } else if sess.target.target.options.is_like_msvc {
        let (cmd, envs) = msvc_link_exe_cmd(sess);
        ("link.exe".to_string(), cmd, envs)
    } else {
        let linker = &sess.target.target.options.linker;
        (linker.clone(), Command::new(&linker), envs)
    }
}

#[cfg(windows)]
pub fn msvc_link_exe_cmd(sess: &Session) -> (Command, Vec<(OsString, OsString)>) {
    use gcc::windows_registry;

    let target = &sess.opts.target_triple;
    let tool = windows_registry::find_tool(target, "link.exe");

    if let Some(tool) = tool {
        let envs = tool.env().to_vec();
        (tool.to_command(), envs)
    } else {
        debug!("Failed to locate linker.");
        (Command::new("link.exe"), vec![])
    }
}

#[cfg(not(windows))]
pub fn msvc_link_exe_cmd(_sess: &Session) -> (Command, Vec<(OsString, OsString)>) {
    (Command::new("link.exe"), vec![])
}

fn command_path(sess: &Session) -> OsString {
    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = sess.host_filesearch(PathKind::All)
                           .get_tools_search_paths();
    if let Some(path) = env::var_os("PATH") {
        new_path.extend(env::split_paths(&path));
    }
    env::join_paths(new_path).unwrap()
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
pub fn link_binary(sess: &Session,
                   trans: &CrateTranslation,
                   outputs: &OutputFilenames,
                   crate_name: &str) -> Vec<PathBuf> {
    let mut out_filenames = Vec::new();
    for &crate_type in sess.crate_types.borrow().iter() {
        // Ignore executable crates if we have -Z no-trans, as they will error.
        if (sess.opts.debugging_opts.no_trans ||
            !sess.opts.output_types.should_trans()) &&
           crate_type == config::CrateTypeExecutable {
            continue;
        }

        if invalid_output_for_target(sess, crate_type) {
           bug!("invalid output type `{:?}` for target os `{}`",
                crate_type, sess.opts.target_triple);
        }
        let mut out_files = link_binary_output(sess, trans, crate_type, outputs, crate_name);
        out_filenames.append(&mut out_files);
    }

    // Remove the temporary object file and metadata if we aren't saving temps
    if !sess.opts.cg.save_temps {
        if sess.opts.output_types.should_trans() {
            for obj in object_filenames(trans, outputs) {
                remove(sess, &obj);
            }
        }
        remove(sess, &outputs.with_extension(METADATA_OBJ_NAME));
        if trans.allocator_module.is_some() {
            remove(sess, &outputs.with_extension(ALLOCATOR_OBJ_NAME));
        }
    }

    out_filenames
}

fn is_writeable(p: &Path) -> bool {
    match p.metadata() {
        Err(..) => true,
        Ok(m) => !m.permissions().readonly()
    }
}

fn filename_for_metadata(sess: &Session, crate_name: &str, outputs: &OutputFilenames) -> PathBuf {
    let out_filename = outputs.single_output_file.clone()
        .unwrap_or(outputs
            .out_directory
            .join(&format!("lib{}{}.rmeta", crate_name, sess.opts.cg.extra_filename)));
    check_file_is_writeable(&out_filename, sess);
    out_filename
}

pub fn each_linked_rlib(sess: &Session,
                        f: &mut FnMut(CrateNum, &Path)) -> Result<(), String> {
    let crates = sess.cstore.used_crates(LinkagePreference::RequireStatic).into_iter();
    let fmts = sess.dependency_formats.borrow();
    let fmts = fmts.get(&config::CrateTypeExecutable)
                   .or_else(|| fmts.get(&config::CrateTypeStaticlib))
                   .or_else(|| fmts.get(&config::CrateTypeCdylib))
                   .or_else(|| fmts.get(&config::CrateTypeProcMacro));
    let fmts = match fmts {
        Some(f) => f,
        None => return Err(format!("could not find formats for rlibs"))
    };
    for (cnum, path) in crates {
        match fmts.get(cnum.as_usize() - 1) {
            Some(&Linkage::NotLinked) |
            Some(&Linkage::IncludedFromDylib) => continue,
            Some(_) => {}
            None => return Err(format!("could not find formats for rlibs"))
        }
        let name = sess.cstore.crate_name(cnum).clone();
        let path = match path {
            LibSource::Some(p) => p,
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
pub fn ignored_for_lto(sess: &Session, cnum: CrateNum) -> bool {
    // `#![no_builtins]` crates don't participate in LTO because the state
    // of builtins gets messed up (our crate isn't tagged with no builtins).
    // Similarly `#![compiler_builtins]` doesn't participate because we want
    // those builtins!
    sess.cstore.is_no_builtins(cnum) || sess.cstore.is_compiler_builtins(cnum)
}

fn out_filename(sess: &Session,
                crate_type: config::CrateType,
                outputs: &OutputFilenames,
                crate_name: &str)
                -> PathBuf {
    let default_filename = filename_for_input(sess, crate_type, crate_name, outputs);
    let out_filename = outputs.outputs.get(&OutputType::Exe)
                              .and_then(|s| s.to_owned())
                              .or_else(|| outputs.single_output_file.clone())
                              .unwrap_or(default_filename);

    check_file_is_writeable(&out_filename, sess);

    out_filename
}

// Make sure files are writeable.  Mac, FreeBSD, and Windows system linkers
// check this already -- however, the Linux linker will happily overwrite a
// read-only file.  We should be consistent.
fn check_file_is_writeable(file: &Path, sess: &Session) {
    if !is_writeable(file) {
        sess.fatal(&format!("output file {} is not writeable -- check its \
                            permissions", file.display()));
    }
}

fn link_binary_output(sess: &Session,
                      trans: &CrateTranslation,
                      crate_type: config::CrateType,
                      outputs: &OutputFilenames,
                      crate_name: &str) -> Vec<PathBuf> {
    let objects = object_filenames(trans, outputs);

    for file in &objects {
        check_file_is_writeable(file, sess);
    }

    let tmpdir = match TempDir::new("rustc") {
        Ok(tmpdir) => tmpdir,
        Err(err) => sess.fatal(&format!("couldn't create a temp dir: {}", err)),
    };

    let mut out_filenames = vec![];

    if outputs.outputs.contains_key(&OutputType::Metadata) {
        let out_filename = filename_for_metadata(sess, crate_name, outputs);
        emit_metadata(sess, trans, &out_filename);
        out_filenames.push(out_filename);
    }

    if outputs.outputs.should_trans() {
        let out_filename = out_filename(sess, crate_type, outputs, crate_name);
        match crate_type {
            config::CrateTypeRlib => {
                link_rlib(sess,
                          trans,
                          RlibFlavor::Normal,
                          &objects,
                          outputs,
                          &out_filename,
                          tmpdir.path()).build();
            }
            config::CrateTypeStaticlib => {
                link_staticlib(sess,
                               trans,
                               outputs,
                               &objects,
                               &out_filename,
                               tmpdir.path());
            }
            _ => {
                link_natively(sess, crate_type, &objects, &out_filename, trans,
                              outputs, tmpdir.path());
            }
        }
        out_filenames.push(out_filename);
    }

    if sess.opts.cg.save_temps {
        let _ = tmpdir.into_path();
    }

    out_filenames
}

fn object_filenames(trans: &CrateTranslation,
                    outputs: &OutputFilenames)
                    -> Vec<PathBuf> {
    trans.modules.iter().map(|module| {
        outputs.temp_path(OutputType::Object, Some(&module.name))
    }).collect()
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

fn emit_metadata<'a>(sess: &'a Session, trans: &CrateTranslation, out_filename: &Path) {
    let result = fs::File::create(out_filename).and_then(|mut f| {
        f.write_all(&trans.metadata.raw_data)
    });

    if let Err(e) = result {
        sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
    }
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
                 trans: &CrateTranslation,
                 flavor: RlibFlavor,
                 objects: &[PathBuf],
                 outputs: &OutputFilenames,
                 out_filename: &Path,
                 tmpdir: &Path) -> ArchiveBuilder<'a> {
    info!("preparing rlib from {:?} to {:?}", objects, out_filename);
    let mut ab = ArchiveBuilder::new(archive_config(sess, out_filename, None));

    for obj in objects {
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
    for lib in sess.cstore.used_libraries() {
        match lib.kind {
            NativeLibraryKind::NativeStatic => {}
            NativeLibraryKind::NativeStaticNobundle |
            NativeLibraryKind::NativeFramework |
            NativeLibraryKind::NativeUnknown => continue,
        }
        ab.add_native_library(&lib.name.as_str());
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
            // contain the metadata in a separate file. We use a temp directory
            // here so concurrent builds in the same directory don't try to use
            // the same filename for metadata (stomping over one another)
            let metadata = tmpdir.join(METADATA_FILENAME);
            emit_metadata(sess, trans, &metadata);
            ab.add_file(&metadata);

            // For LTO purposes, the bytecode of this library is also inserted
            // into the archive.  If codegen_units > 1, we insert each of the
            // bitcode files.
            for obj in objects {
                // Note that we make sure that the bytecode filename in the
                // archive is never exactly 16 bytes long by adding a 16 byte
                // extension to it. This is to work around a bug in LLDB that
                // would cause it to crash if the name of a file in an archive
                // was exactly 16 bytes.
                let bc_filename = obj.with_extension("bc");
                let bc_deflated_filename = tmpdir.join({
                    obj.with_extension("bytecode.deflate").file_name().unwrap()
                });

                let mut bc_data = Vec::new();
                match fs::File::open(&bc_filename).and_then(|mut f| {
                    f.read_to_end(&mut bc_data)
                }) {
                    Ok(..) => {}
                    Err(e) => sess.fatal(&format!("failed to read bytecode: {}",
                                                 e))
                }

                let mut bc_data_deflated = Vec::new();
                DeflateEncoder::new(&mut bc_data_deflated, Compression::Fast)
                    .write_all(&bc_data).unwrap();

                let mut bc_file_deflated = match fs::File::create(&bc_deflated_filename) {
                    Ok(file) => file,
                    Err(e) => {
                        sess.fatal(&format!("failed to create compressed \
                                             bytecode file: {}", e))
                    }
                };

                match write_rlib_bytecode_object_v1(&mut bc_file_deflated,
                                                    &bc_data_deflated) {
                    Ok(()) => {}
                    Err(e) => {
                        sess.fatal(&format!("failed to write compressed \
                                             bytecode: {}", e));
                    }
                };

                ab.add_file(&bc_deflated_filename);

                // See the bottom of back::write::run_passes for an explanation
                // of when we do and don't keep .#module-name#.bc files around.
                let user_wants_numbered_bitcode =
                        sess.opts.output_types.contains_key(&OutputType::Bitcode) &&
                        sess.opts.cg.codegen_units > 1;
                if !sess.opts.cg.save_temps && !user_wants_numbered_bitcode {
                    remove(sess, &bc_filename);
                }
            }

            // After adding all files to the archive, we need to update the
            // symbol table of the archive. This currently dies on macOS (see
            // #11162), and isn't necessary there anyway
            if !sess.target.target.options.is_like_osx {
                ab.update_symbols();
            }
        }

        RlibFlavor::StaticlibBase => {
            if trans.allocator_module.is_some() {
                ab.add_file(&outputs.with_extension(ALLOCATOR_OBJ_NAME));
            }
        }
    }

    ab
}

fn write_rlib_bytecode_object_v1(writer: &mut Write,
                                 bc_data_deflated: &[u8]) -> io::Result<()> {
    let bc_data_deflated_size: u64 = bc_data_deflated.len() as u64;

    writer.write_all(RLIB_BYTECODE_OBJECT_MAGIC)?;
    writer.write_all(&[1, 0, 0, 0])?;
    writer.write_all(&[
        (bc_data_deflated_size >>  0) as u8,
        (bc_data_deflated_size >>  8) as u8,
        (bc_data_deflated_size >> 16) as u8,
        (bc_data_deflated_size >> 24) as u8,
        (bc_data_deflated_size >> 32) as u8,
        (bc_data_deflated_size >> 40) as u8,
        (bc_data_deflated_size >> 48) as u8,
        (bc_data_deflated_size >> 56) as u8,
    ])?;
    writer.write_all(&bc_data_deflated)?;

    let number_of_bytes_written_so_far =
        RLIB_BYTECODE_OBJECT_MAGIC.len() +                // magic id
        mem::size_of_val(&RLIB_BYTECODE_OBJECT_VERSION) + // version
        mem::size_of_val(&bc_data_deflated_size) +        // data size field
        bc_data_deflated_size as usize;                    // actual data

    // If the number of bytes written to the object so far is odd, add a
    // padding byte to make it even. This works around a crash bug in LLDB
    // (see issue #15950)
    if number_of_bytes_written_so_far % 2 == 1 {
        writer.write_all(&[0])?;
    }

    return Ok(());
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
                  trans: &CrateTranslation,
                  outputs: &OutputFilenames,
                  objects: &[PathBuf],
                  out_filename: &Path,
                  tempdir: &Path) {
    let mut ab = link_rlib(sess,
                           trans,
                           RlibFlavor::StaticlibBase,
                           objects,
                           outputs,
                           out_filename,
                           tempdir);
    let mut all_native_libs = vec![];

    let res = each_linked_rlib(sess, &mut |cnum, path| {
        let name = sess.cstore.crate_name(cnum);
        let native_libs = sess.cstore.native_libraries(cnum);

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
                    sess.lto() && !ignored_for_lto(sess, cnum),
                    skip_object_files).unwrap();

        all_native_libs.extend(sess.cstore.native_libraries(cnum));
    });
    if let Err(e) = res {
        sess.fatal(&e);
    }

    ab.update_symbols();
    ab.build();

    if !all_native_libs.is_empty() {
        if sess.opts.prints.contains(&PrintRequest::NativeStaticLibs) {
            print_native_static_libs(sess, &all_native_libs);
        } else {
            // Fallback for backwards compatibility only
            print_native_static_libs_legacy(sess, &all_native_libs);
        }
    }
}

fn print_native_static_libs_legacy(sess: &Session, all_native_libs: &[NativeLibrary]) {
    sess.note_without_error("link against the following native artifacts when linking against \
                             this static library");
    sess.note_without_error("This list will not be printed by default. \
        Please add --print=native-static-libs if you need this information");

    for lib in all_native_libs.iter().filter(|l| relevant_lib(sess, l)) {
        let name = match lib.kind {
            NativeLibraryKind::NativeStaticNobundle |
            NativeLibraryKind::NativeUnknown => "library",
            NativeLibraryKind::NativeFramework => "framework",
            // These are included, no need to print them
            NativeLibraryKind::NativeStatic => continue,
        };
        sess.note_without_error(&format!("{}: {}", name, lib.name));
    }
}

fn print_native_static_libs(sess: &Session, all_native_libs: &[NativeLibrary]) {
    let lib_args: Vec<_> = all_native_libs.iter()
        .filter(|l| relevant_lib(sess, l))
        .filter_map(|lib| match lib.kind {
            NativeLibraryKind::NativeStaticNobundle |
            NativeLibraryKind::NativeUnknown => {
                if sess.target.target.options.is_like_msvc {
                    Some(format!("{}.lib", lib.name))
                } else {
                    Some(format!("-l{}", lib.name))
                }
            },
            NativeLibraryKind::NativeFramework => {
                // ld-only syntax, since there are no frameworks in MSVC
                Some(format!("-framework {}", lib.name))
            },
            // These are included, no need to print them
            NativeLibraryKind::NativeStatic => None,
        })
        .collect();
    if !lib_args.is_empty() {
        sess.note_without_error("Link against the following native artifacts when linking \
                                 against this static library. The order and any duplication \
                                 can be significant on some platforms.");
        // Prefix for greppability
        sess.note_without_error(format!("native-static-libs: {}", &lib_args.join(" ")));
    }
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively(sess: &Session,
                 crate_type: config::CrateType,
                 objects: &[PathBuf],
                 out_filename: &Path,
                 trans: &CrateTranslation,
                 outputs: &OutputFilenames,
                 tmpdir: &Path) {
    info!("preparing {:?} from {:?} to {:?}", crate_type, objects, out_filename);
    let flavor = sess.linker_flavor();

    // The invocations of cc share some flags across platforms
    let (pname, mut cmd, envs) = get_linker(sess);
    // This will set PATH on windows
    cmd.envs(envs);

    let root = sess.target_filesearch(PathKind::Native).get_lib_path();
    if let Some(args) = sess.target.target.options.pre_link_args.get(&flavor) {
        cmd.args(args);
    }
    if let Some(ref args) = sess.opts.debugging_opts.pre_link_args {
        cmd.args(args);
    }
    cmd.args(&sess.opts.debugging_opts.pre_link_arg);

    let pre_link_objects = if crate_type == config::CrateTypeExecutable {
        &sess.target.target.options.pre_link_objects_exe
    } else {
        &sess.target.target.options.pre_link_objects_dll
    };
    for obj in pre_link_objects {
        cmd.arg(root.join(obj));
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
        let mut linker = trans.linker_info.to_linker(cmd, &sess);
        link_args(&mut *linker, sess, crate_type, tmpdir,
                  objects, out_filename, outputs, trans);
        cmd = linker.finalize();
    }
    if let Some(args) = sess.target.target.options.late_link_args.get(&flavor) {
        cmd.args(args);
    }
    for obj in &sess.target.target.options.post_link_objects {
        cmd.arg(root.join(obj));
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
        prog = time(sess.time_passes(), "running linker", || cmd.output());
        if !retry_on_segfault || i > 3 {
            break
        }
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
        let msg = "clang: error: unable to execute command: \
                   Segmentation fault: 11";
        if !out.contains(msg) {
            break
        }

        sess.struct_warn("looks like the linker segfaulted when we tried to \
                          call it, automatically retrying again")
            .note(&format!("{:?}", cmd))
            .note(&out)
            .emit();
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
                                         pname,
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
            sess.struct_err(&format!("could not exec the linker `{}`: {}", pname, e))
                .note(&format!("{:?}", &cmd))
                .emit();
            if sess.target.target.options.is_like_msvc && e.kind() == io::ErrorKind::NotFound {
                sess.note_without_error("the msvc targets depend on the msvc linker \
                    but `link.exe` was not found");
                sess.note_without_error("please ensure that VS 2013 or VS 2015 was installed \
                    with the Visual C++ option");
            }
            sess.abort_if_errors();
        }
    }


    // On macOS, debuggers need this utility to get run to do some munging of
    // the symbols
    if sess.target.target.options.is_like_osx && sess.opts.debuginfo != NoDebugInfo {
        match Command::new("dsymutil").arg(out_filename).output() {
            Ok(..) => {}
            Err(e) => sess.fatal(&format!("failed to run dsymutil: {}", e)),
        }
    }
}

fn link_args(cmd: &mut Linker,
             sess: &Session,
             crate_type: config::CrateType,
             tmpdir: &Path,
             objects: &[PathBuf],
             out_filename: &Path,
             outputs: &OutputFilenames,
             trans: &CrateTranslation) {

    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = sess.target_filesearch(PathKind::All).get_lib_path();

    // target descriptor
    let t = &sess.target.target;

    cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));
    for obj in objects {
        cmd.add_object(obj);
    }
    cmd.output_filename(out_filename);

    if crate_type == config::CrateTypeExecutable &&
       sess.target.target.options.is_like_windows {
        if let Some(ref s) = trans.windows_subsystem {
            cmd.subsystem(s);
        }
    }

    // If we're building a dynamic library then some platforms need to make sure
    // that all symbols are exported correctly from the dynamic library.
    if crate_type != config::CrateTypeExecutable ||
       sess.target.target.options.is_like_emscripten {
        cmd.export_symbols(tmpdir, crate_type);
    }

    // When linking a dynamic library, we put the metadata into a section of the
    // executable. This metadata is in a separate object file from the main
    // object file, so we link that in here.
    if crate_type == config::CrateTypeDylib ||
       crate_type == config::CrateTypeProcMacro {
        cmd.add_object(&outputs.with_extension(METADATA_OBJ_NAME));
    }

    if trans.allocator_module.is_some() {
        cmd.add_object(&outputs.with_extension(ALLOCATOR_OBJ_NAME));
    }

    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    if !sess.opts.cg.link_dead_code {
        let keep_metadata = crate_type == config::CrateTypeDylib;
        cmd.gc_sections(keep_metadata);
    }

    let used_link_args = sess.cstore.used_link_args();

    if crate_type == config::CrateTypeExecutable &&
       t.options.position_independent_executables {
        let empty_vec = Vec::new();
        let args = sess.opts.cg.link_args.as_ref().unwrap_or(&empty_vec);
        let more_args = &sess.opts.cg.link_arg;
        let mut args = args.iter().chain(more_args.iter()).chain(used_link_args.iter());

        if get_reloc_model(sess) == llvm::RelocMode::PIC
            && !args.any(|x| *x == "-static") {
            cmd.position_independent_executable();
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
        RelroLevel::Off => {},
    }

    // Pass optimization flags down to the linker.
    cmd.optimize();

    // Pass debuginfo flags down to the linker.
    cmd.debuginfo();

    // We want to prevent the compiler from accidentally leaking in any system
    // libraries, so we explicitly ask gcc to not link to any libraries by
    // default. Note that this does not happen for windows because windows pulls
    // in some large number of libraries and I couldn't quite figure out which
    // subset we wanted.
    if t.options.no_default_libraries {
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
    add_local_native_libraries(cmd, sess);
    add_upstream_rust_crates(cmd, sess, crate_type, tmpdir);
    add_upstream_native_libraries(cmd, sess, crate_type);

    // # Telling the linker what we're doing

    if crate_type != config::CrateTypeExecutable {
        cmd.build_dylib(out_filename);
    }

    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // addl_lib_search_paths
    if sess.opts.cg.rpath {
        let sysroot = sess.sysroot();
        let target_triple = &sess.opts.target_triple;
        let mut get_install_prefix_lib_path = || {
            let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");
            let tlib = filesearch::relative_target_lib_path(sysroot, target_triple);
            let mut path = PathBuf::from(install_prefix);
            path.push(&tlib);

            path
        };
        let mut rpath_config = RPathConfig {
            used_crates: sess.cstore.used_crates(LinkagePreference::RequireDynamic),
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
fn add_local_native_libraries(cmd: &mut Linker, sess: &Session) {
    sess.target_filesearch(PathKind::All).for_each_lib_search_path(|path, k| {
        match k {
            PathKind::Framework => { cmd.framework_path(path); }
            _ => { cmd.include_path(&fix_windows_verbatim_for_gcc(path)); }
        }
    });

    let relevant_libs = sess.cstore.used_libraries().into_iter().filter(|l| {
        relevant_lib(sess, l)
    });

    let search_path = archive_search_paths(sess);
    for lib in relevant_libs {
        match lib.kind {
            NativeLibraryKind::NativeUnknown => cmd.link_dylib(&lib.name.as_str()),
            NativeLibraryKind::NativeFramework => cmd.link_framework(&lib.name.as_str()),
            NativeLibraryKind::NativeStaticNobundle => cmd.link_staticlib(&lib.name.as_str()),
            NativeLibraryKind::NativeStatic => cmd.link_whole_staticlib(&lib.name.as_str(),
                                                                        &search_path)
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
fn add_upstream_rust_crates(cmd: &mut Linker,
                            sess: &Session,
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
    let deps = sess.cstore.used_crates(LinkagePreference::RequireDynamic);

    let mut compiler_builtins = None;

    for &(cnum, _) in &deps {
        // We may not pass all crates through to the linker. Some crates may
        // appear statically in an existing dylib, meaning we'll pick up all the
        // symbols from the dylib.
        let src = sess.cstore.used_crate_source(cnum);
        match data[cnum.as_usize() - 1] {
            _ if sess.cstore.is_profiler_runtime(cnum) => {
                add_static_crate(cmd, sess, tmpdir, crate_type, cnum);
            }
            _ if sess.cstore.is_sanitizer_runtime(cnum) => {
                link_sanitizer_runtime(cmd, sess, tmpdir, cnum);
            }
            // compiler-builtins are always placed last to ensure that they're
            // linked correctly.
            _ if sess.cstore.is_compiler_builtins(cnum) => {
                assert!(compiler_builtins.is_none());
                compiler_builtins = Some(cnum);
            }
            Linkage::NotLinked |
            Linkage::IncludedFromDylib => {}
            Linkage::Static => {
                add_static_crate(cmd, sess, tmpdir, crate_type, cnum);
            }
            Linkage::Dynamic => {
                add_dynamic_crate(cmd, sess, &src.dylib.unwrap().0)
            }
        }
    }

    // compiler-builtins are always placed last to ensure that they're
    // linked correctly.
    // We must always link the `compiler_builtins` crate statically. Even if it
    // was already "included" in a dylib (e.g. `libstd` when `-C prefer-dynamic`
    // is used)
    if let Some(cnum) = compiler_builtins {
        add_static_crate(cmd, sess, tmpdir, crate_type, cnum);
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
    fn link_sanitizer_runtime(cmd: &mut Linker,
                              sess: &Session,
                              tmpdir: &Path,
                              cnum: CrateNum) {
        let src = sess.cstore.used_crate_source(cnum);
        let cratepath = &src.rlib.unwrap().0;

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
            if f.ends_with("bytecode.deflate") || f == METADATA_FILENAME {
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
    fn add_static_crate(cmd: &mut Linker,
                        sess: &Session,
                        tmpdir: &Path,
                        crate_type: config::CrateType,
                        cnum: CrateNum) {
        let src = sess.cstore.used_crate_source(cnum);
        let cratepath = &src.rlib.unwrap().0;

        // See the comment above in `link_staticlib` and `link_rlib` for why if
        // there's a static library that's not relevant we skip all object
        // files.
        let native_libs = sess.cstore.native_libraries(cnum);
        let skip_native = native_libs.iter().any(|lib| {
            lib.kind == NativeLibraryKind::NativeStatic && !relevant_lib(sess, lib)
        });

        if (!sess.lto() || ignored_for_lto(sess, cnum)) &&
           crate_type != config::CrateTypeDylib &&
           !skip_native {
            cmd.link_rlib(&fix_windows_verbatim_for_gcc(cratepath));
            return
        }

        let dst = tmpdir.join(cratepath.file_name().unwrap());
        let name = cratepath.file_name().unwrap().to_str().unwrap();
        let name = &name[3..name.len() - 5]; // chop off lib/.rlib

        time(sess.time_passes(), &format!("altering {}.rlib", name), || {
            let cfg = archive_config(sess, &dst, Some(cratepath));
            let mut archive = ArchiveBuilder::new(cfg);
            archive.update_symbols();

            let mut any_objects = false;
            for f in archive.src_files() {
                if f.ends_with("bytecode.deflate") || f == METADATA_FILENAME {
                    archive.remove_file(&f);
                    continue
                }

                let canonical = f.replace("-", "_");
                let canonical_name = name.replace("-", "_");

                let is_rust_object =
                    canonical.starts_with(&canonical_name) && {
                        let num = &f[name.len()..f.len() - 2];
                        num.len() > 0 && num[1..].parse::<u32>().is_ok()
                    };

                // If we've been requested to skip all native object files
                // (those not generated by the rust compiler) then we can skip
                // this file. See above for why we may want to do this.
                let skip_because_cfg_say_so = skip_native && !is_rust_object;

                // If we're performing LTO and this is a rust-generated object
                // file, then we don't need the object file as it's part of the
                // LTO module. Note that `#![no_builtins]` is excluded from LTO,
                // though, so we let that object file slide.
                let skip_because_lto = sess.lto() && is_rust_object &&
                                        !sess.cstore.is_no_builtins(cnum);

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
            if crate_type == config::CrateTypeDylib &&
               !sess.cstore.is_compiler_builtins(cnum) {
                cmd.link_whole_rlib(&fix_windows_verbatim_for_gcc(&dst));
            } else {
                cmd.link_rlib(&fix_windows_verbatim_for_gcc(&dst));
            }
        });
    }

    // Same thing as above, but for dynamic crates instead of static crates.
    fn add_dynamic_crate(cmd: &mut Linker, sess: &Session, cratepath: &Path) {
        // If we're performing LTO, then it should have been previously required
        // that all upstream rust dependencies were available in an rlib format.
        assert!(!sess.lto());

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
fn add_upstream_native_libraries(cmd: &mut Linker, sess: &Session, crate_type: config::CrateType) {
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

    let crates = sess.cstore.used_crates(LinkagePreference::RequireStatic);
    for (cnum, _) in crates {
        for lib in sess.cstore.native_libraries(cnum) {
            if !relevant_lib(sess, &lib) {
                continue
            }
            match lib.kind {
                NativeLibraryKind::NativeUnknown => cmd.link_dylib(&lib.name.as_str()),
                NativeLibraryKind::NativeFramework => cmd.link_framework(&lib.name.as_str()),
                NativeLibraryKind::NativeStaticNobundle => {
                    // Link "static-nobundle" native libs only if the crate they originate from
                    // is being linked statically to the current crate.  If it's linked dynamically
                    // or is an rlib already included via some other dylib crate, the symbols from
                    // native libs will have already been included in that dylib.
                    if data[cnum.as_usize() - 1] == Linkage::Static {
                        cmd.link_staticlib(&lib.name.as_str())
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
