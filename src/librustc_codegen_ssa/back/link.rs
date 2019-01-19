/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.

use rustc::session::{Session, config};
use rustc::session::search_paths::PathKind;
use rustc::middle::dependency_format::Linkage;
use rustc::middle::cstore::LibSource;
use rustc_target::spec::LinkerFlavor;
use rustc::hir::def_id::CrateNum;

use super::command::Command;
use CrateInfo;

use cc::windows_registry;
use std::fs;
use std::path::{Path, PathBuf};
use std::env;

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
                let stem = if linker.extension().and_then(|ext| ext.to_str()) == Some("exe") {
                    linker.file_stem().and_then(|stem| stem.to_str())
                } else {
                    linker.to_str()
                }.unwrap_or_else(|| {
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
