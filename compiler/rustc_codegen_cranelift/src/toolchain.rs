//! Locating various executables part of a C toolchain.

use std::path::PathBuf;

use rustc_middle::bug;
use rustc_session::Session;
use rustc_target::spec::LinkerFlavor;

/// Tries to infer the path of a binary for the target toolchain from the linker name.
pub(crate) fn get_toolchain_binary(sess: &Session, tool: &str) -> PathBuf {
    let (mut linker, _linker_flavor) = linker_and_flavor(sess);
    let linker_file_name = linker
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| sess.fatal("couldn't extract file name from specified linker"));

    if linker_file_name == "ld.lld" {
        if tool != "ld" {
            linker.set_file_name(tool)
        }
    } else {
        let tool_file_name = linker_file_name
            .replace("ld", tool)
            .replace("gcc", tool)
            .replace("clang", tool)
            .replace("cc", tool);

        linker.set_file_name(tool_file_name)
    }

    linker
}

// Adapted from https://github.com/rust-lang/rust/blob/5db778affee7c6600c8e7a177c48282dab3f6292/src/librustc_codegen_ssa/back/link.rs#L848-L931
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
    if let Some(ret) = infer_from(
        sess,
        sess.opts.cg.linker.clone(),
        sess.opts.cg.linker_flavor,
    ) {
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
