use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_fs_util::fix_windows_verbatim_for_gcc;
use rustc_hir::def_id::CrateNum;
use rustc_middle::middle::cstore::{EncodedMetadata, LibSource, NativeLib};
use rustc_middle::middle::dependency_format::Linkage;
use rustc_session::config::{self, CFGuard, CrateType, DebugInfo};
use rustc_session::config::{OutputFilenames, OutputType, PrintRequest, SanitizerSet};
use rustc_session::output::{check_file_is_writeable, invalid_output_for_target, out_filename};
use rustc_session::search_paths::PathKind;
use rustc_session::utils::NativeLibKind;
/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.
use rustc_session::{filesearch, Session};
use rustc_span::symbol::Symbol;
use rustc_target::spec::crt_objects::{CrtObjects, CrtObjectsFallback};
use rustc_target::spec::{LinkOutputKind, LinkerFlavor, LldFlavor};
use rustc_target::spec::{PanicStrategy, RelocModel, RelroLevel, Target};

use super::archive::ArchiveBuilder;
use super::command::Command;
use super::linker::{self, Linker};
use super::rpath::{self, RPathConfig};
use crate::{looks_like_rust_object_file, CodegenResults, CrateInfo, METADATA_FILENAME};

use cc::windows_registry;
use tempfile::Builder as TempFileBuilder;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};
use std::{ascii, char, env, fmt, fs, io, mem, str};

pub fn remove(sess: &Session, path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        sess.err(&format!("failed to remove {}: {}", path.display(), e));
    }
}

/// Performs the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    outputs: &OutputFilenames,
    crate_name: &str,
    target_cpu: &str,
) {
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

        if outputs.outputs.should_codegen() {
            let tmpdir = TempFileBuilder::new()
                .prefix("rustc")
                .tempdir()
                .unwrap_or_else(|err| sess.fatal(&format!("couldn't create a temp dir: {}", err)));
            let path = MaybeTempDir::new(tmpdir, sess.opts.cg.save_temps);
            let out_filename = out_filename(sess, crate_type, outputs, crate_name);
            match crate_type {
                CrateType::Rlib => {
                    let _timer = sess.timer("link_rlib");
                    link_rlib::<B>(sess, codegen_results, RlibFlavor::Normal, &out_filename, &path)
                        .build();
                }
                CrateType::Staticlib => {
                    link_staticlib::<B>(sess, codegen_results, &out_filename, &path);
                }
                _ => {
                    link_natively::<B>(
                        sess,
                        crate_type,
                        &out_filename,
                        codegen_results,
                        path.as_ref(),
                        target_cpu,
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
            if sess.opts.output_types.should_codegen()
                && !preserve_objects_for_their_debuginfo(sess)
            {
                for obj in codegen_results.modules.iter().filter_map(|m| m.object.as_ref()) {
                    remove(sess, obj);
                }
            }
            if let Some(ref metadata_module) = codegen_results.metadata_module {
                if let Some(ref obj) = metadata_module.object {
                    remove(sess, obj);
                }
            }
            if let Some(ref allocator_module) = codegen_results.allocator_module {
                if let Some(ref obj) = allocator_module.object {
                    remove(sess, obj);
                }
            }
        }
    });
}

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
fn get_linker(
    sess: &Session,
    linker: &Path,
    flavor: LinkerFlavor,
    self_contained: bool,
) -> Command {
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
            LinkerFlavor::Msvc if sess.opts.cg.linker.is_none() && sess.target.linker.is_none() => {
                Command::new(msvc_tool.as_ref().map(|t| t.path()).unwrap_or(linker))
            }
            _ => Command::new(linker),
        },
    };

    // UWP apps have API restrictions enforced during Store submissions.
    // To comply with the Windows App Certification Kit,
    // MSVC needs to link with the Store versions of the runtime libraries (vcruntime, msvcrt, etc).
    let t = &sess.target;
    if (flavor == LinkerFlavor::Msvc || flavor == LinkerFlavor::Lld(LldFlavor::Link))
        && t.vendor == "uwp"
    {
        if let Some(ref tool) = msvc_tool {
            let original_path = tool.path();
            if let Some(ref root_lib_path) = original_path.ancestors().nth(4) {
                let arch = match t.arch.as_str() {
                    "x86_64" => Some("x64".to_string()),
                    "x86" => Some("x86".to_string()),
                    "aarch64" => Some("arm64".to_string()),
                    "arm" => Some("arm".to_string()),
                    _ => None,
                };
                if let Some(ref a) = arch {
                    // FIXME: Move this to `fn linker_with_args`.
                    let mut arg = OsString::from("/LIBPATH:");
                    arg.push(format!("{}\\lib\\{}\\store", root_lib_path.display(), a.to_string()));
                    cmd.arg(&arg);
                } else {
                    warn!("arch is not supported");
                }
            } else {
                warn!("MSVC root path lib location not found");
            }
        } else {
            warn!("link.exe not found");
        }
    }

    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = sess.host_filesearch(PathKind::All).get_tools_search_paths(self_contained);
    let mut msvc_changed_path = false;
    if sess.target.is_like_msvc {
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

    cmd
}

pub fn each_linked_rlib(
    info: &CrateInfo,
    f: &mut dyn FnMut(CrateNum, &Path),
) -> Result<(), String> {
    let crates = info.used_crates_static.iter();
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
    for &(cnum, ref path) in crates {
        match fmts.get(cnum.as_usize() - 1) {
            Some(&Linkage::NotLinked | &Linkage::IncludedFromDylib) => continue,
            Some(_) => {}
            None => return Err("could not find formats for rlibs".to_string()),
        }
        let name = &info.crate_name[&cnum];
        let path = match *path {
            LibSource::Some(ref p) => p,
            LibSource::MetadataOnly => {
                return Err(format!(
                    "could not find rlib for: `{}`, found rmeta (metadata) file",
                    name
                ));
            }
            LibSource::None => return Err(format!("could not find rlib for: `{}`", name)),
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
pub fn emit_metadata(sess: &Session, metadata: &EncodedMetadata, tmpdir: &MaybeTempDir) -> PathBuf {
    let out_filename = tmpdir.as_ref().join(METADATA_FILENAME);
    let result = fs::write(&out_filename, &metadata.raw_data);

    if let Err(e) = result {
        sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
    }

    out_filename
}

// Create an 'rlib'
//
// An rlib in its current incarnation is essentially a renamed .a file. The
// rlib primarily contains the object file of the crate, but it also contains
// all of the object files from native libraries. This is done by unzipping
// native libraries and inserting all of the contents into this archive.
fn link_rlib<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    flavor: RlibFlavor,
    out_filename: &Path,
    tmpdir: &MaybeTempDir,
) -> B {
    info!("preparing rlib to {:?}", out_filename);
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
            NativeLibKind::StaticBundle => {}
            NativeLibKind::StaticNoBundle
            | NativeLibKind::Dylib
            | NativeLibKind::Framework
            | NativeLibKind::RawDylib
            | NativeLibKind::Unspecified => continue,
        }
        if let Some(name) = lib.name {
            ab.add_native_library(name);
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
            ab.add_file(&emit_metadata(sess, &codegen_results.metadata, tmpdir));

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
fn link_staticlib<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    out_filename: &Path,
    tempdir: &MaybeTempDir,
) {
    let mut ab =
        link_rlib::<B>(sess, codegen_results, RlibFlavor::StaticlibBase, out_filename, tempdir);
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
        let skip_object_files = native_libs
            .iter()
            .any(|lib| lib.kind == NativeLibKind::StaticBundle && !relevant_lib(sess, lib));
        ab.add_rlib(
            path,
            &name.as_str(),
            are_upstream_rust_objects_already_included(sess)
                && !ignored_for_lto(sess, &codegen_results.crate_info, cnum),
            skip_object_files,
        )
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
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively<'a, B: ArchiveBuilder<'a>>(
    sess: &'a Session,
    crate_type: CrateType,
    out_filename: &Path,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
    target_cpu: &str,
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
        target_cpu,
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

        // Check to see if the link failed with "unrecognized command line option:
        // '-no-pie'" for gcc or "unknown argument: '-no-pie'" for clang. If so,
        // reperform the link step without the -no-pie option. This is safe because
        // if the linker doesn't support -no-pie then it should not default to
        // linking executables as pie. Different versions of gcc seem to use
        // different quotes in the error message so don't check for them.
        if sess.target.linker_is_gnu
            && flavor != LinkerFlavor::Ld
            && (out.contains("unrecognized command line option")
                || out.contains("unknown argument"))
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
            && (out.contains("unrecognized command line option")
                || out.contains("unknown argument"))
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
                "looks like the linker segfaulted when we tried to call it, \
                 automatically retrying again. cmd = {:?}, out = {}.",
                cmd, out,
            );
            continue;
        }

        if is_illegal_instruction(&output.status) {
            warn!(
                "looks like the linker hit an illegal instruction when we \
                 tried to call it, automatically retrying again. cmd = {:?}, ]\
                 out = {}, status = {}.",
                cmd, out, output.status,
            );
            continue;
        }

        #[cfg(unix)]
        fn is_illegal_instruction(status: &ExitStatus) -> bool {
            use std::os::unix::prelude::*;
            status.signal() == Some(libc::SIGILL)
        }

        #[cfg(windows)]
        fn is_illegal_instruction(_status: &ExitStatus) -> bool {
            false
        }
    }

    match prog {
        Ok(prog) => {
            fn escape_string(s: &[u8]) -> String {
                str::from_utf8(s).map(|s| s.to_owned()).unwrap_or_else(|_| {
                    let mut x = "Non-UTF-8 output: ".to_string();
                    x.extend(s.iter().flat_map(|&b| ascii::escape_default(b)).map(char::from));
                    x
                })
            }
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                sess.struct_err(&format!(
                    "linking with `{}` failed: {}",
                    linker_path.display(),
                    prog.status
                ))
                .note(&format!("{:?}", &cmd))
                .note(&escape_string(&output))
                .emit();

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
            info!("linker stderr:\n{}", escape_string(&prog.stderr));
            info!("linker stdout:\n{}", escape_string(&prog.stdout));
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

    // On macOS, debuggers need this utility to get run to do some munging of
    // the symbols. Note, though, that if the object files are being preserved
    // for their debug information there's no need for us to run dsymutil.
    if sess.target.is_like_osx
        && sess.opts.debuginfo != DebugInfo::None
        && !preserve_objects_for_their_debuginfo(sess)
    {
        if let Err(e) = Command::new("dsymutil").arg(out_filename).output() {
            sess.fatal(&format!("failed to run dsymutil: {}", e))
        }
    }
}

fn link_sanitizers(sess: &Session, crate_type: CrateType, linker: &mut dyn Linker) {
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
}

fn link_sanitizer_runtime(sess: &Session, linker: &mut dyn Linker, name: &str) {
    let default_sysroot = filesearch::get_or_default_sysroot();
    let default_tlib =
        filesearch::make_target_lib_path(&default_sysroot, sess.opts.target_triple.triple());
    let channel = option_env!("CFG_RELEASE_CHANNEL")
        .map(|channel| format!("-{}", channel))
        .unwrap_or_default();

    match sess.opts.target_triple.triple() {
        "x86_64-apple-darwin" => {
            // On Apple platforms, the sanitizer is always built as a dylib, and
            // LLVM will link to `@rpath/*.dylib`, so we need to specify an
            // rpath to the library as well (the rpath should be absolute, see
            // PR #41352 for details).
            let libname = format!("rustc{}_rt.{}", channel, name);
            let rpath = default_tlib.to_str().expect("non-utf8 component in path");
            linker.args(&["-Wl,-rpath", "-Xlinker", rpath]);
            linker.link_dylib(Symbol::intern(&libname));
        }
        "aarch64-fuchsia"
        | "aarch64-unknown-linux-gnu"
        | "x86_64-fuchsia"
        | "x86_64-unknown-freebsd"
        | "x86_64-unknown-linux-gnu" => {
            let filename = format!("librustc{}_rt.{}.a", channel, name);
            let path = default_tlib.join(&filename);
            linker.link_whole_rlib(&path);
        }
        _ => {}
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

fn linker_and_flavor(sess: &Session) -> (PathBuf, LinkerFlavor) {
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
    if sess.target.is_like_osx {
        return !sess.opts.debugging_opts.run_dsymutil;
    }

    false
}

pub fn archive_search_paths(sess: &Session) -> Vec<PathBuf> {
    sess.target_filesearch(PathKind::Native).search_path_dirs()
}

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
                NativeLibKind::StaticNoBundle
                | NativeLibKind::Dylib
                | NativeLibKind::Unspecified => {
                    if sess.target.is_like_msvc {
                        Some(format!("{}.lib", name))
                    } else {
                        Some(format!("-l{}", name))
                    }
                }
                NativeLibKind::Framework => {
                    // ld-only syntax, since there are no frameworks in MSVC
                    Some(format!("-framework {}", name))
                }
                // These are included, no need to print them
                NativeLibKind::StaticBundle | NativeLibKind::RawDylib => None,
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

    #[cfg(unix)]
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
        (CrateType::Executable, false, RelocModel::Pic) => LinkOutputKind::DynamicPicExe,
        (CrateType::Executable, false, _) => LinkOutputKind::DynamicNoPicExe,
        (CrateType::Executable, true, RelocModel::Pic) => LinkOutputKind::StaticPicExe,
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

/// Add arbitrary "user defined" args defined from command line and by `#[link_args]` attributes.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_user_defined_link_args(
    cmd: &mut dyn Linker,
    sess: &Session,
    codegen_results: &CodegenResults,
) {
    cmd.args(&sess.opts.cg.link_args);
    cmd.args(&*codegen_results.crate_info.link_args);
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

/// Link native libraries corresponding to the current crate and all libraries corresponding to
/// all its dependency crates.
/// FIXME: Consider combining this with the functions above adding object files for the local crate.
fn link_local_crate_native_libs_and_dependent_crate_libs<'a, B: ArchiveBuilder<'a>>(
    cmd: &mut dyn Linker,
    sess: &'a Session,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
) {
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
    // Upstream rust libraries are not allowed to depend on our local native
    // libraries as that would violate the structure of the DAG, in that
    // scenario they are required to link to them as well in a shared fashion.
    //
    // Note that upstream rust libraries may contain native dependencies as
    // well, but they also can't depend on what we just started to add to the
    // link line. And finally upstream native libraries can't depend on anything
    // in this DAG so far because they're only dylibs and dylibs can only depend
    // on other dylibs (e.g., other native deps).
    //
    // If -Zlink-native-libraries=false is set, then the assumption is that an
    // external build system already has the native dependencies defined, and it
    // will provide them to the linker itself.
    if sess.opts.debugging_opts.link_native_libraries {
        add_local_native_libraries(cmd, sess, codegen_results);
    }
    add_upstream_rust_crates::<B>(cmd, sess, codegen_results, crate_type, tmpdir);
    if sess.opts.debugging_opts.link_native_libraries {
        add_upstream_native_libraries(cmd, sess, codegen_results, crate_type);
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
    // addl_lib_search_paths
    if sess.opts.cg.rpath {
        let target_triple = sess.opts.target_triple.triple();
        let mut get_install_prefix_lib_path = || {
            let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");
            let tlib = filesearch::relative_target_lib_path(&sess.sysroot, target_triple);
            let mut path = PathBuf::from(install_prefix);
            path.push(&tlib);

            path
        };
        let mut rpath_config = RPathConfig {
            used_crates: &codegen_results.crate_info.used_crates_dynamic,
            out_filename: out_filename.to_path_buf(),
            has_rpath: sess.target.has_rpath,
            is_like_osx: sess.target.is_like_osx,
            linker_is_gnu: sess.target.linker_is_gnu,
            get_install_prefix_lib_path: &mut get_install_prefix_lib_path,
        };
        cmd.args(&rpath::get_rpath_flags(&mut rpath_config));
    }
}

/// Produce the linker command line containing linker path and arguments.
/// `NO-OPT-OUT` marks the arguments that cannot be removed from the command line
/// by the user without creating a custom target specification.
/// `OBJECT-FILES` specify whether the arguments can add object files.
/// `CUSTOMIZATION-POINT` means that arbitrary arguments defined by the user
/// or by the target spec can be inserted here.
/// `AUDIT-ORDER` - need to figure out whether the option is order-dependent or not.
fn linker_with_args<'a, B: ArchiveBuilder<'a>>(
    path: &Path,
    flavor: LinkerFlavor,
    sess: &'a Session,
    crate_type: CrateType,
    tmpdir: &Path,
    out_filename: &Path,
    codegen_results: &CodegenResults,
    target_cpu: &str,
) -> Command {
    let crt_objects_fallback = crt_objects_fallback(sess, crate_type);
    let base_cmd = get_linker(sess, path, flavor, crt_objects_fallback);
    // FIXME: Move `/LIBPATH` addition for uwp targets from the linker construction
    // to the linker args construction.
    assert!(base_cmd.get_args().is_empty() || sess.target.vendor == "uwp");
    let cmd = &mut *codegen_results.linker_info.to_linker(base_cmd, &sess, flavor, target_cpu);
    let link_output_kind = link_output_kind(sess, crate_type);

    // NO-OPT-OUT, OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_pre_link_args(cmd, sess, flavor);

    // NO-OPT-OUT, OBJECT-FILES-NO
    add_apple_sdk(cmd, sess, flavor);

    // NO-OPT-OUT
    add_link_script(cmd, sess, tmpdir, crate_type);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    if sess.target.is_like_fuchsia && crate_type == CrateType::Executable {
        let prefix = if sess.opts.debugging_opts.sanitizer.contains(SanitizerSet::ADDRESS) {
            "asan/"
        } else {
            ""
        };
        cmd.arg(format!("--dynamic-linker={}ld.so.1", prefix));
    }

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    if sess.target.eh_frame_header {
        cmd.add_eh_frame_header();
    }

    // NO-OPT-OUT, OBJECT-FILES-NO
    if crt_objects_fallback {
        cmd.no_crt_objects();
    }

    // NO-OPT-OUT, OBJECT-FILES-YES
    add_pre_link_objects(cmd, sess, link_output_kind, crt_objects_fallback);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    if sess.target.is_like_emscripten {
        cmd.arg("-s");
        cmd.arg(if sess.panic_strategy() == PanicStrategy::Abort {
            "DISABLE_EXCEPTION_CATCHING=1"
        } else {
            "DISABLE_EXCEPTION_CATCHING=0"
        });
    }

    // OBJECT-FILES-YES, AUDIT-ORDER
    link_sanitizers(sess, crate_type, cmd);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // Linker plugins should be specified early in the list of arguments
    // FIXME: How "early" exactly?
    cmd.linker_plugin_lto();

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    // FIXME: Order-dependent, at least relatively to other args adding searh directories.
    add_library_search_dirs(cmd, sess, crt_objects_fallback);

    // OBJECT-FILES-YES
    add_local_crate_regular_objects(cmd, codegen_results);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    cmd.output_filename(out_filename);

    // OBJECT-FILES-NO, AUDIT-ORDER
    if crate_type == CrateType::Executable && sess.target.is_like_windows {
        if let Some(ref s) = codegen_results.windows_subsystem {
            cmd.subsystem(s);
        }
    }

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    // If we're building something like a dynamic library then some platforms
    // need to make sure that all symbols are exported correctly from the
    // dynamic library.
    cmd.export_symbols(tmpdir, crate_type);

    // OBJECT-FILES-YES
    add_local_crate_metadata_objects(cmd, crate_type, codegen_results);

    // OBJECT-FILES-YES
    add_local_crate_allocator_objects(cmd, codegen_results);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // FIXME: Order dependent, applies to the following objects. Where should it be placed?
    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    if !sess.link_dead_code() {
        let keep_metadata = crate_type == CrateType::Dylib;
        cmd.gc_sections(keep_metadata);
    }

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    cmd.set_output_kind(link_output_kind, out_filename);

    // OBJECT-FILES-NO, AUDIT-ORDER
    add_relro_args(cmd, sess);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // Pass optimization flags down to the linker.
    cmd.optimize();

    // OBJECT-FILES-NO, AUDIT-ORDER
    // Pass debuginfo and strip flags down to the linker.
    cmd.debuginfo(sess.opts.debugging_opts.strip);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // We want to prevent the compiler from accidentally leaking in any system libraries,
    // so by default we tell linkers not to link to any default libraries.
    if !sess.opts.cg.default_linker_libraries && sess.target.no_default_libraries {
        cmd.no_default_libraries();
    }

    // OBJECT-FILES-YES
    link_local_crate_native_libs_and_dependent_crate_libs::<B>(
        cmd,
        sess,
        crate_type,
        codegen_results,
        tmpdir,
    );

    // OBJECT-FILES-NO, AUDIT-ORDER
    if sess.opts.cg.profile_generate.enabled() || sess.opts.debugging_opts.instrument_coverage {
        cmd.pgo_gen();
    }

    // OBJECT-FILES-NO, AUDIT-ORDER
    if sess.opts.cg.control_flow_guard != CFGuard::Disabled {
        cmd.control_flow_guard();
    }

    // OBJECT-FILES-NO, AUDIT-ORDER
    add_rpath_args(cmd, sess, codegen_results, out_filename);

    // OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_user_defined_link_args(cmd, sess, codegen_results);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    cmd.finalize();

    // NO-OPT-OUT, OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_late_link_args(cmd, sess, flavor, crate_type, codegen_results);

    // NO-OPT-OUT, OBJECT-FILES-YES
    add_post_link_objects(cmd, sess, link_output_kind, crt_objects_fallback);

    // NO-OPT-OUT, OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_post_link_args(cmd, sess, flavor);

    cmd.take_cmd()
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

    let search_path = archive_search_paths(sess);
    for lib in relevant_libs {
        let name = match lib.name {
            Some(l) => l,
            None => continue,
        };
        match lib.kind {
            NativeLibKind::Dylib | NativeLibKind::Unspecified => cmd.link_dylib(name),
            NativeLibKind::Framework => cmd.link_framework(name),
            NativeLibKind::StaticNoBundle => cmd.link_staticlib(name),
            NativeLibKind::StaticBundle => cmd.link_whole_staticlib(name, &search_path),
            NativeLibKind::RawDylib => {
                // FIXME(#58713): Proper handling for raw dylibs.
                bug!("raw_dylib feature not yet implemented");
            }
        }
    }
}

// # Rust Crate linking
//
// Rust crates are not considered at all when creating an rlib output. All
// dependencies will be linked when producing the final output (instead of
// the intermediate rlib version)
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
    // Crates available for linking thus far.
    let mut available = FxHashSet::default();
    // Crates required to satisfy dependencies discovered so far.
    let mut required = FxHashSet::default();

    let info = &codegen_results.crate_info;
    for &(cnum, _) in deps.iter().rev() {
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

        // See the comment above in `link_staticlib` and `link_rlib` for why if
        // there's a static library that's not relevant we skip all object
        // files.
        let native_libs = &codegen_results.crate_info.native_libraries[&cnum];
        let skip_native = native_libs
            .iter()
            .any(|lib| lib.kind == NativeLibKind::StaticBundle && !relevant_lib(sess, lib));

        if (!are_upstream_rust_objects_already_included(sess)
            || ignored_for_lto(sess, &codegen_results.crate_info, cnum))
            && crate_type != CrateType::Dylib
            && !skip_native
        {
            cmd.link_rlib(&fix_windows_verbatim_for_gcc(cratepath));
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

            // If we're creating a dylib, then we need to include the
            // whole of each object in our archive into that artifact. This is
            // because a `dylib` can be reused as an intermediate artifact.
            //
            // Note, though, that we don't want to include the whole of a
            // compiler-builtins crate (e.g., compiler-rt) because it'll get
            // repeatedly linked anyway.
            if crate_type == CrateType::Dylib
                && codegen_results.crate_info.compiler_builtins != Some(cnum)
            {
                cmd.link_whole_rlib(&fix_windows_verbatim_for_gcc(&dst));
            } else {
                cmd.link_rlib(&fix_windows_verbatim_for_gcc(&dst));
            }
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
            parent.unwrap_or(Path::new("")),
        );
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
fn add_upstream_native_libraries(
    cmd: &mut dyn Linker,
    sess: &Session,
    codegen_results: &CodegenResults,
    crate_type: CrateType,
) {
    // Be sure to use a topological sorting of crates because there may be
    // interdependencies between native libraries. When passing -nodefaultlibs,
    // for example, almost all native libraries depend on libc, so we have to
    // make sure that's all the way at the right (liblibc is near the base of
    // the dependency chain).
    //
    // This passes RequireStatic, but the actual requirement doesn't matter,
    // we're just getting an ordering of crate numbers, we're not worried about
    // the paths.
    let (_, data) = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| *ty == crate_type)
        .expect("failed to find crate type in dependency format list");

    let crates = &codegen_results.crate_info.used_crates_static;
    for &(cnum, _) in crates {
        for lib in codegen_results.crate_info.native_libraries[&cnum].iter() {
            let name = match lib.name {
                Some(l) => l,
                None => continue,
            };
            if !relevant_lib(sess, &lib) {
                continue;
            }
            match lib.kind {
                NativeLibKind::Dylib | NativeLibKind::Unspecified => cmd.link_dylib(name),
                NativeLibKind::Framework => cmd.link_framework(name),
                NativeLibKind::StaticNoBundle => {
                    // Link "static-nobundle" native libs only if the crate they originate from
                    // is being linked statically to the current crate.  If it's linked dynamically
                    // or is an rlib already included via some other dylib crate, the symbols from
                    // native libs will have already been included in that dylib.
                    if data[cnum.as_usize() - 1] == Linkage::Static {
                        cmd.link_staticlib(name)
                    }
                }
                // ignore statically included native libraries here as we've
                // already included them when we included the rust library
                // previously
                NativeLibKind::StaticBundle => {}
                NativeLibKind::RawDylib => {
                    // FIXME(#58713): Proper handling for raw dylibs.
                    bug!("raw_dylib feature not yet implemented");
                }
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
        ("aarch64", "ios") => "iphoneos",
        ("x86", "ios") => "iphonesimulator",
        ("x86_64", "ios") if llvm_target.contains("macabi") => "macosx10.15",
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
    let arch_name = llvm_target.split('-').next().expect("LLVM target must have a hyphen");
    cmd.args(&["-arch", arch_name, "-isysroot", &sdk_root, "-Wl,-syslibroot", &sdk_root]);
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
