//! C-compiler probing and detection.
//!
//! This module will fill out the `cc` and `cxx` maps of `Build` by looking for
//! C and C++ compilers for each target configured. A compiler is found through
//! a number of vectors (in order of precedence)
//!
//! 1. Configuration via `target.$target.cc` in `bootstrap.toml`.
//! 2. Configuration via `target.$target.android-ndk` in `bootstrap.toml`, if
//!    applicable
//! 3. Special logic to probe on OpenBSD
//! 4. The `CC_$target` environment variable.
//! 5. The `CC` environment variable.
//! 6. "cc"
//!
//! Some of this logic is implemented here, but much of it is farmed out to the
//! `cc` crate itself, so we end up having the same fallbacks as there.
//! Similar logic is then used to find a C++ compiler, just some s/cc/c++/ is
//! used.
//!
//! It is intended that after this module has run no C/C++ compiler will
//! ever be probed for. Instead the compilers found here will be used for
//! everything.

use std::collections::HashSet;
use std::iter;
use std::path::{Path, PathBuf};

use crate::core::config::flags::Subcommand;
use crate::core::config::{CompressDebuginfo, TargetSelection};
use crate::core::session::{CLang, GitRepo, Session};
use crate::utils::exec::{BootstrapCommand, command};

/// Creates and configures a new [`cc::Build`] instance for the given target.
fn new_cc_build(sess: &Session, target: TargetSelection) -> cc::Build {
    let mut cfg = cc::Build::new();
    cfg.cargo_metadata(false)
        .opt_level(2)
        .warnings(false)
        .debug(false)
        // We have to configure out_dir, otherwise flag_if_supported will not work
        .out_dir(sess.tempdir().join("cc-rs-out-dir"))
        .target(&target.triple)
        .host(&sess.host_target.triple);

    match sess.config.compress_debuginfo(target) {
        CompressDebuginfo::Zlib => {
            cfg.flag_if_supported("-gz");
        }
        CompressDebuginfo::Off => {}
    }

    match sess.crt_static(target) {
        Some(a) => {
            cfg.static_crt(a);
        }
        None => {
            if target.is_msvc() {
                cfg.static_crt(true);
            }
        }
    }
    cfg
}

/// Probes for C and C++ compilers and configures the corresponding entries in the [`Session`]
/// structure.
///
/// This function determines which targets need a C compiler (and, if needed, a C++ compiler)
/// by combining the primary build target, host targets, and any additional targets. For
/// each target, it calls [`fill_target_compiler`] to configure the necessary compiler tools.
pub(crate) fn fill_compilers(sess: &mut Session) {
    let mut targets: HashSet<_> = match sess.config.cmd {
        // We don't need to check cross targets for these commands.
        Subcommand::Clean { .. }
        | Subcommand::Check { .. }
        | Subcommand::Format { .. }
        | Subcommand::Setup { .. } => {
            sess.hosts.iter().cloned().chain(iter::once(sess.host_target)).collect()
        }

        _ => {
            // For all targets we're going to need a C compiler for building some shims
            // and such as well as for being a linker for Rust code.
            sess.targets
                .iter()
                .chain(&sess.hosts)
                .cloned()
                .chain(iter::once(sess.host_target))
                .collect()
        }
    };

    // When we intend to build wasm proc macros, we'll need to detect a toolchain for linking those
    // as well. In the future it would be good to make this a no-op given that we shouldn't need to
    // build any C/C++ code for wasm...
    if sess.config.wasm_proc_macros {
        targets.insert(TargetSelection::from_user("wasm32-unknown-unknown"));
    }

    for target in targets {
        fill_target_compiler(sess, target);
    }
}

/// Probes and configures the C and C++ compilers for a single target.
///
/// This function uses both user-specified configuration (from `bootstrap.toml`) and auto-detection
/// logic to determine the correct C/C++ compilers for the target. It also determines the appropriate
/// archiver (`ar`) and sets up additional compilation flags (both handled and unhandled).
fn fill_target_compiler(sess: &mut Session, target: TargetSelection) {
    let mut cfg = new_cc_build(sess, target);
    let config = sess.config.target_config.get(&target);
    if let Some(cc) = config
        .and_then(|c| c.cc.clone())
        .or_else(|| default_compiler(&cfg, Language::C, target, sess))
    {
        cfg.compiler(cc);
    }

    let compiler = cfg.get_compiler();
    let ar = config
        .and_then(|c| c.ar.clone())
        .or_else(|| cfg.try_get_archiver().map(|c| PathBuf::from(c.get_program())).ok());

    sess.cc.insert(target, compiler.clone());
    let mut cflags = sess.cc_handled_cflags(target, CLang::C);
    cflags.extend(sess.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::C));

    // If we use llvm-libunwind, we will need a C++ compiler as well for all targets
    // We'll need one anyways if the target triple is also a host triple
    let mut cfg = new_cc_build(sess, target);
    cfg.cpp(true);
    let cxx_configured = if let Some(cxx) = config
        .and_then(|c| c.cxx.clone())
        .or_else(|| default_compiler(&cfg, Language::CPlusPlus, target, sess))
    {
        cfg.compiler(cxx);
        true
    } else {
        // Use an auto-detected compiler (or one configured via `CXX_target_triple` env vars).
        cfg.try_get_compiler().is_ok()
    };

    // for VxWorks, record CXX compiler which will be used in lib.rs:linker()
    if cxx_configured || target.contains("vxworks") {
        let compiler = cfg.get_compiler();
        sess.cxx.insert(target, compiler);
    }

    sess.do_if_verbose(|| println!("CC_{} = {:?}", target.triple, sess.cc(target)));
    sess.do_if_verbose(|| println!("CFLAGS_{} = {cflags:?}", target.triple));
    if let Ok(cxx) = sess.cxx(target) {
        let mut cxxflags = sess.cc_handled_cflags(target, CLang::Cxx);
        cxxflags.extend(sess.cc_unhandled_cflags(target, GitRepo::Rustc, CLang::Cxx));
        sess.do_if_verbose(|| println!("CXX_{} = {cxx:?}", target.triple));
        sess.do_if_verbose(|| println!("CXXFLAGS_{} = {cxxflags:?}", target.triple));
    }
    if let Some(ar) = ar {
        sess.do_if_verbose(|| println!("AR_{} = {ar:?}", target.triple));
        sess.ar.insert(target, ar);
    }

    if let Some(ranlib) = config.and_then(|c| c.ranlib.clone()) {
        sess.ranlib.insert(target, ranlib);
    }
}

/// Determines the default compiler for a given target and language when not explicitly
/// configured in `bootstrap.toml`.
fn default_compiler(
    cfg: &cc::Build,
    compiler: Language,
    target: TargetSelection,
    sess: &Session,
) -> Option<PathBuf> {
    match &*target.triple {
        // When compiling for android we may have the NDK configured in the
        // bootstrap.toml in which case we look there. Otherwise the default
        // compiler already takes into account the triple in question.
        t if t.contains("android") => {
            sess.config.android_ndk.as_ref().map(|ndk| ndk_compiler(compiler, &target.triple, ndk))
        }

        // The default gcc version from OpenBSD may be too old, try using egcc,
        // which is a gcc version from ports, if this is the case.
        t if t.contains("openbsd") => {
            let c = cfg.get_compiler();
            let gnu_compiler = compiler.gcc();
            if !c.path().ends_with(gnu_compiler) {
                return None;
            }

            let mut cmd = BootstrapCommand::from(c.to_command());
            let output = cmd.arg("--version").run_capture_stdout(sess).stdout();
            let i = output.find(" 4.")?;
            match output[i + 3..].chars().next().unwrap() {
                '0'..='6' => {}
                _ => return None,
            }
            let alternative = format!("e{gnu_compiler}");
            if command(&alternative).run_capture(sess).is_success() {
                Some(PathBuf::from(alternative))
            } else {
                None
            }
        }

        "mips-unknown-linux-musl" if compiler == Language::C => {
            if cfg.get_compiler().path().to_str() == Some("gcc") {
                Some(PathBuf::from("mips-linux-musl-gcc"))
            } else {
                None
            }
        }
        "mipsel-unknown-linux-musl" if compiler == Language::C => {
            if cfg.get_compiler().path().to_str() == Some("gcc") {
                Some(PathBuf::from("mipsel-linux-musl-gcc"))
            } else {
                None
            }
        }

        t if t.contains("musl") && compiler == Language::C => {
            if let Some(root) = sess.musl_root(target) {
                let guess = root.join("bin/musl-gcc");
                if guess.exists() { Some(guess) } else { None }
            } else {
                None
            }
        }

        t if t.contains("-wasi") => {
            let root = if let Some(path) = sess.wasi_sdk_path.as_ref() {
                path
            } else {
                if sess.config.is_running_on_ci() {
                    panic!("ERROR: WASI_SDK_PATH must be configured for a -wasi target on CI");
                }
                println!("WARNING: WASI_SDK_PATH not set, using default cc/cxx compiler");
                return None;
            };
            let compiler = match compiler {
                Language::C => format!("{t}-clang"),
                Language::CPlusPlus => format!("{t}-clang++"),
            };
            let compiler = root.join("bin").join(compiler);
            Some(compiler)
        }

        _ => None,
    }
}

/// Constructs the path to the Android NDK compiler for the given target triple and language.
///
/// This helper function transform the target triple by converting certain architecture names
/// (for example, translating "arm" to "arm7a"), appends the minimum API level (hardcoded as "21"
/// for NDK r26d), and then constructs the full path based on the provided NDK directory and host
/// platform.
pub(crate) fn ndk_compiler(compiler: Language, triple: &str, ndk: &Path) -> PathBuf {
    let mut triple_iter = triple.split('-');
    let triple_translated = if let Some(arch) = triple_iter.next() {
        let arch_new = match arch {
            "arm" | "armv7" | "armv7neon" | "thumbv7" | "thumbv7neon" => "armv7a",
            other => other,
        };
        std::iter::once(arch_new).chain(triple_iter).collect::<Vec<&str>>().join("-")
    } else {
        triple.to_string()
    };

    // The earliest API supported by NDK r26d is 21.
    let api_level = "21";
    let compiler = format!("{}{}-{}", triple_translated, api_level, compiler.clang());
    let host_tag = if cfg!(target_os = "macos") {
        // The NDK uses universal binaries, so this is correct even on ARM.
        "darwin-x86_64"
    } else if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else {
        // NDK r26d only has official releases for macOS, Windows and Linux.
        // Try the Linux directory everywhere else, on the assumption that the OS has an
        // emulation layer that can cope (e.g. BSDs).
        "linux-x86_64"
    };
    ndk.join("toolchains").join("llvm").join("prebuilt").join(host_tag).join("bin").join(compiler)
}

/// Representing the target programming language for a native compiler.
///
/// This enum is used to indicate whether a particular compiler is intended for C or C++.
/// It also provides helper methods for obtaining the standard executable names for GCC and
/// clang-based compilers.
#[derive(PartialEq)]
pub(crate) enum Language {
    /// The compiler is targeting C.
    C,
    /// The compiler is targeting C++.
    CPlusPlus,
}

impl Language {
    /// Returns the executable name for a GCC compiler corresponding to this language.
    fn gcc(self) -> &'static str {
        match self {
            Language::C => "gcc",
            Language::CPlusPlus => "g++",
        }
    }

    /// Returns the executable name for a clang-based compiler corresponding to this language.
    fn clang(self) -> &'static str {
        match self {
            Language::C => "clang",
            Language::CPlusPlus => "clang++",
        }
    }
}

#[cfg(test)]
mod tests;
