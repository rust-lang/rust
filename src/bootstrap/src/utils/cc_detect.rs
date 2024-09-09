//! C-compiler probing and detection.
//!
//! This module will fill out the `cc` and `cxx` maps of `Build` by looking for
//! C and C++ compilers for each target configured. A compiler is found through
//! a number of vectors (in order of precedence)
//!
//! 1. Configuration via `target.$target.cc` in `config.toml`.
//! 2. Configuration via `target.$target.android-ndk` in `config.toml`, if
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
use std::path::{Path, PathBuf};
use std::{env, iter};

use crate::core::config::TargetSelection;
use crate::utils::exec::{command, BootstrapCommand};
use crate::{Build, CLang, GitRepo};

// The `cc` crate doesn't provide a way to obtain a path to the detected archiver,
// so use some simplified logic here. First we respect the environment variable `AR`, then
// try to infer the archiver path from the C compiler path.
// In the future this logic should be replaced by calling into the `cc` crate.
fn cc2ar(cc: &Path, target: TargetSelection) -> Option<PathBuf> {
    if let Some(ar) = env::var_os(format!("AR_{}", target.triple.replace('-', "_"))) {
        Some(PathBuf::from(ar))
    } else if let Some(ar) = env::var_os("AR") {
        Some(PathBuf::from(ar))
    } else if target.is_msvc() {
        None
    } else if target.contains("musl") || target.contains("openbsd") {
        Some(PathBuf::from("ar"))
    } else if target.contains("vxworks") {
        Some(PathBuf::from("wr-ar"))
    } else if target.contains("android") || target.contains("-wasi") {
        Some(cc.parent().unwrap().join(PathBuf::from("llvm-ar")))
    } else {
        let parent = cc.parent().unwrap();
        let file = cc.file_name().unwrap().to_str().unwrap();
        for suffix in &["gcc", "cc", "clang"] {
            if let Some(idx) = file.rfind(suffix) {
                let mut file = file[..idx].to_owned();
                file.push_str("ar");
                return Some(parent.join(&file));
            }
        }
        Some(parent.join(file))
    }
}

fn new_cc_build(build: &Build, target: TargetSelection) -> cc::Build {
    let mut cfg = cc::Build::new();
    cfg.cargo_metadata(false)
        .opt_level(2)
        .warnings(false)
        .debug(false)
        // Compress debuginfo
        .flag_if_supported("-gz")
        .target(&target.triple)
        .host(&build.build.triple);
    match build.crt_static(target) {
        Some(a) => {
            cfg.static_crt(a);
        }
        None => {
            if target.is_msvc() {
                cfg.static_crt(true);
            }
            if target.contains("musl") {
                cfg.static_flag(true);
            }
        }
    }
    cfg
}

pub fn find(build: &Build) {
    let targets: HashSet<_> = match build.config.cmd {
        // We don't need to check cross targets for these commands.
        crate::Subcommand::Clean { .. }
        | crate::Subcommand::Suggest { .. }
        | crate::Subcommand::Format { .. }
        | crate::Subcommand::Setup { .. } => {
            build.hosts.iter().cloned().chain(iter::once(build.build)).collect()
        }

        _ => {
            // For all targets we're going to need a C compiler for building some shims
            // and such as well as for being a linker for Rust code.
            build
                .targets
                .iter()
                .chain(&build.hosts)
                .cloned()
                .chain(iter::once(build.build))
                .collect()
        }
    };

    for target in targets.into_iter() {
        find_target(build, target);
    }
}

pub fn find_target(build: &Build, target: TargetSelection) {
    let mut cfg = new_cc_build(build, target);
    let config = build.config.target_config.get(&target);
    if let Some(cc) = config
        .and_then(|c| c.cc.clone())
        .or_else(|| default_compiler(&mut cfg, Language::C, target, build))
    {
        cfg.compiler(cc);
    }

    let compiler = cfg.get_compiler();
    let ar = if let ar @ Some(..) = config.and_then(|c| c.ar.clone()) {
        ar
    } else {
        cc2ar(compiler.path(), target)
    };

    build.cc.borrow_mut().insert(target, compiler.clone());
    let cflags = build.cflags(target, GitRepo::Rustc, CLang::C);

    // If we use llvm-libunwind, we will need a C++ compiler as well for all targets
    // We'll need one anyways if the target triple is also a host triple
    let mut cfg = new_cc_build(build, target);
    cfg.cpp(true);
    let cxx_configured = if let Some(cxx) = config
        .and_then(|c| c.cxx.clone())
        .or_else(|| default_compiler(&mut cfg, Language::CPlusPlus, target, build))
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
        build.cxx.borrow_mut().insert(target, compiler);
    }

    build.verbose(|| println!("CC_{} = {:?}", target.triple, build.cc(target)));
    build.verbose(|| println!("CFLAGS_{} = {cflags:?}", target.triple));
    if let Ok(cxx) = build.cxx(target) {
        let cxxflags = build.cflags(target, GitRepo::Rustc, CLang::Cxx);
        build.verbose(|| println!("CXX_{} = {cxx:?}", target.triple));
        build.verbose(|| println!("CXXFLAGS_{} = {cxxflags:?}", target.triple));
    }
    if let Some(ar) = ar {
        build.verbose(|| println!("AR_{} = {ar:?}", target.triple));
        build.ar.borrow_mut().insert(target, ar);
    }

    if let Some(ranlib) = config.and_then(|c| c.ranlib.clone()) {
        build.ranlib.borrow_mut().insert(target, ranlib);
    }
}

fn default_compiler(
    cfg: &mut cc::Build,
    compiler: Language,
    target: TargetSelection,
    build: &Build,
) -> Option<PathBuf> {
    match &*target.triple {
        // When compiling for android we may have the NDK configured in the
        // config.toml in which case we look there. Otherwise the default
        // compiler already takes into account the triple in question.
        t if t.contains("android") => {
            build.config.android_ndk.as_ref().map(|ndk| ndk_compiler(compiler, &target.triple, ndk))
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
            let output = cmd.arg("--version").run_capture_stdout(build).stdout();
            let i = output.find(" 4.")?;
            match output[i + 3..].chars().next().unwrap() {
                '0'..='6' => {}
                _ => return None,
            }
            let alternative = format!("e{gnu_compiler}");
            if command(&alternative).run_capture(build).is_success() {
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
            if let Some(root) = build.musl_root(target) {
                let guess = root.join("bin/musl-gcc");
                if guess.exists() { Some(guess) } else { None }
            } else {
                None
            }
        }

        t if t.contains("-wasi") => {
            let root = PathBuf::from(std::env::var_os("WASI_SDK_PATH")?);
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

/// The target programming language for a native compiler.
#[derive(PartialEq)]
pub(crate) enum Language {
    /// The compiler is targeting C.
    C,
    /// The compiler is targeting C++.
    CPlusPlus,
}

impl Language {
    /// Obtains the name of a compiler in the GCC collection.
    fn gcc(self) -> &'static str {
        match self {
            Language::C => "gcc",
            Language::CPlusPlus => "g++",
        }
    }

    /// Obtains the name of a compiler in the clang suite.
    fn clang(self) -> &'static str {
        match self {
            Language::C => "clang",
            Language::CPlusPlus => "clang++",
        }
    }
}
