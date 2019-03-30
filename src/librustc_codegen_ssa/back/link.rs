/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.

use rustc::session::{Session, config};
use rustc::session::search_paths::PathKind;
use rustc::middle::dependency_format::Linkage;
use rustc::middle::cstore::{LibSource, NativeLibrary, NativeLibraryKind};
use rustc_target::spec::LinkerFlavor;
use rustc::hir::def_id::CrateNum;
use rustc_fs_util::fix_windows_verbatim_for_gcc;

use super::command::Command;
use crate::{CrateInfo, CodegenResults};
use crate::back::linker::Linker;

use cc::windows_registry;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::env;

pub use rustc_codegen_utils::link::*;

pub fn remove(sess: &Session, path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        sess.err(&format!("failed to remove {}: {}",
                          path.display(),
                          e));
    }
}

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

pub fn each_linked_rlib(sess: &Session,
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
pub fn ignored_for_lto(sess: &Session, info: &CrateInfo, cnum: CrateNum) -> bool {
    // If our target enables builtin function lowering in LLVM then the
    // crates providing these functions don't participate in LTO (e.g.
    // no_builtins or compiler builtins crates).
    !sess.target.target.options.no_builtins &&
        (info.compiler_builtins == Some(cnum) || info.is_no_builtins.contains(&cnum))
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
                LinkerFlavor::PtxLinker => "rust-ptx-linker",
            }), flavor)),
            (Some(linker), None) => {
                let stem = linker
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_else(|| {
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
    if let Some(ret) = infer_from(sess, sess.opts.cg.linker.clone(), sess.opts.cg.linker_flavor) {
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

/// Returns a boolean indicating whether we should preserve the object files on
/// the filesystem for their debug information. This is often useful with
/// split-dwarf like schemes.
pub fn preserve_objects_for_their_debuginfo(sess: &Session) -> bool {
    // If the objects don't have debuginfo there's nothing to preserve.
    if sess.opts.debuginfo == config::DebugInfo::None {
        return false
    }

    // If we're only producing artifacts that are archives, no need to preserve
    // the objects as they're losslessly contained inside the archives.
    let output_linked = sess.crate_types.borrow()
        .iter()
        .any(|&x| x != config::CrateType::Rlib && x != config::CrateType::Staticlib);
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

pub fn archive_search_paths(sess: &Session) -> Vec<PathBuf> {
    sess.target_filesearch(PathKind::Native).search_path_dirs()
}

enum RlibFlavor {
    Normal,
    StaticlibBase,
}

pub fn print_native_static_libs(sess: &Session, all_native_libs: &[NativeLibrary]) {
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

pub fn get_file_path(sess: &Session, name: &str) -> PathBuf {
    let fs = sess.target_filesearch(PathKind::Native);
    let file_path = fs.get_lib_path().join(name);
    if file_path.exists() {
        return file_path
    }
    for search_path in fs.search_paths() {
        let file_path = search_path.dir.join(name);
        if file_path.exists() {
            return file_path
        }
    }
    PathBuf::from(name)
}

pub fn exec_linker(sess: &Session, cmd: &mut Command, out_filename: &Path, tmpdir: &Path)
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
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                        '\\' | ' ' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
            }
            Ok(())
        }
    }
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
pub fn add_local_native_libraries(cmd: &mut dyn Linker,
                              sess: &Session,
                              codegen_results: &CodegenResults) {
    let filesearch = sess.target_filesearch(PathKind::All);
    for search_path in filesearch.search_paths() {
        match search_path.kind {
            PathKind::Framework => { cmd.framework_path(&search_path.dir); }
            _ => { cmd.include_path(&fix_windows_verbatim_for_gcc(&search_path.dir)); }
        }
    }

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
pub fn add_upstream_native_libraries(cmd: &mut dyn Linker,
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

pub fn relevant_lib(sess: &Session, lib: &NativeLibrary) -> bool {
    match lib.cfg {
        Some(ref cfg) => syntax::attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

pub fn are_upstream_rust_objects_already_included(sess: &Session) -> bool {
    match sess.lto() {
        config::Lto::Fat => true,
        config::Lto::Thin => {
            // If we defer LTO to the linker, we haven't run LTO ourselves, so
            // any upstream object files have not been copied yet.
            !sess.opts.cg.linker_plugin_lto.enabled()
        }
        config::Lto::No |
        config::Lto::ThinLocal => false,
    }
}
