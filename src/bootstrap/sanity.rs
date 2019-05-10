//! Sanity checking performed by rustbuild before actually executing anything.
//!
//! This module contains the implementation of ensuring that the build
//! environment looks reasonable before progressing. This will verify that
//! various programs like git and python exist, along with ensuring that all C
//! compilers for cross-compiling are found.
//!
//! In theory if we get past this phase it's a bug if a build fails, but in
//! practice that's likely not true!

use std::collections::HashMap;
use std::env;
use std::ffi::{OsString, OsStr};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use build_helper::{output, t};

use crate::Build;

struct Finder {
    cache: HashMap<OsString, Option<PathBuf>>,
    path: OsString,
}

impl Finder {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            path: env::var_os("PATH").unwrap_or_default()
        }
    }

    fn maybe_have<S: AsRef<OsStr>>(&mut self, cmd: S) -> Option<PathBuf> {
        let cmd: OsString = cmd.as_ref().into();
        let path = &self.path;
        self.cache.entry(cmd.clone()).or_insert_with(|| {
            for path in env::split_paths(path) {
                let target = path.join(&cmd);
                let mut cmd_exe = cmd.clone();
                cmd_exe.push(".exe");

                if target.is_file()                   // some/path/git
                    || path.join(&cmd_exe).exists()   // some/path/git.exe
                    || target.join(&cmd_exe).exists() // some/path/git/git.exe
                {
                    return Some(target);
                }
            }
            None
        }).clone()
    }

    fn must_have<S: AsRef<OsStr>>(&mut self, cmd: S) -> PathBuf {
        self.maybe_have(&cmd).unwrap_or_else(|| {
            panic!("\n\ncouldn't find required command: {:?}\n\n", cmd.as_ref());
        })
    }
}

pub fn check(build: &mut Build) {
    let path = env::var_os("PATH").unwrap_or_default();
    // On Windows, quotes are invalid characters for filename paths, and if
    // one is present as part of the PATH then that can lead to the system
    // being unable to identify the files properly. See
    // https://github.com/rust-lang/rust/issues/34959 for more details.
    if cfg!(windows) && path.to_string_lossy().contains('\"') {
        panic!("PATH contains invalid character '\"'");
    }

    let mut cmd_finder = Finder::new();
    // If we've got a git directory we're gonna need git to update
    // submodules and learn about various other aspects.
    if build.rust_info.is_git() {
        cmd_finder.must_have("git");
    }

    // We need cmake, but only if we're actually building LLVM or sanitizers.
    let building_llvm = build.hosts.iter()
        .filter_map(|host| build.config.target_config.get(host))
        .any(|config| config.llvm_config.is_none());
    if building_llvm || build.config.sanitizers {
        cmd_finder.must_have("cmake");
    }

    // Ninja is currently only used for LLVM itself.
    if building_llvm {
        if build.config.ninja {
            // Some Linux distros rename `ninja` to `ninja-build`.
            // CMake can work with either binary name.
            if cmd_finder.maybe_have("ninja-build").is_none() {
                cmd_finder.must_have("ninja");
            }
        }

        // If ninja isn't enabled but we're building for MSVC then we try
        // doubly hard to enable it. It was realized in #43767 that the msbuild
        // CMake generator for MSVC doesn't respect configuration options like
        // disabling LLVM assertions, which can often be quite important!
        //
        // In these cases we automatically enable Ninja if we find it in the
        // environment.
        if !build.config.ninja && build.config.build.contains("msvc") {
            if cmd_finder.maybe_have("ninja").is_some() {
                build.config.ninja = true;
            }
        }
    }

    build.config.python = build.config.python.take().map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("python2.7"))
        .or_else(|| cmd_finder.maybe_have("python2"))
        .or_else(|| env::var_os("BOOTSTRAP_PYTHON").map(PathBuf::from)) // set by bootstrap.py
        .or_else(|| Some(cmd_finder.must_have("python")));

    build.config.nodejs = build.config.nodejs.take().map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("node"))
        .or_else(|| cmd_finder.maybe_have("nodejs"));

    build.config.gdb = build.config.gdb.take().map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("gdb"));

    // We're gonna build some custom C code here and there, host triples
    // also build some C++ shims for LLVM so we need a C++ compiler.
    for target in &build.targets {
        // On emscripten we don't actually need the C compiler to just
        // build the target artifacts, only for testing. For the sake
        // of easier bot configuration, just skip detection.
        if target.contains("emscripten") {
            continue;
        }

        // We don't use a C compiler on wasm32
        if target.contains("wasm32") {
            continue;
        }

        if !build.config.dry_run {
            cmd_finder.must_have(build.cc(*target));
            if let Some(ar) = build.ar(*target) {
                cmd_finder.must_have(ar);
            }
        }
    }

    for host in &build.hosts {
        if !build.config.dry_run {
            cmd_finder.must_have(build.cxx(*host).unwrap());
        }
    }

    // Externally configured LLVM requires FileCheck to exist
    let filecheck = build.llvm_filecheck(build.build);
    if !filecheck.starts_with(&build.out) && !filecheck.exists() && build.config.codegen_tests {
        panic!("FileCheck executable {:?} does not exist", filecheck);
    }

    for target in &build.targets {
        // Can't compile for iOS unless we're on macOS
        if target.contains("apple-ios") &&
           !build.build.contains("apple-darwin") {
            panic!("the iOS target is only supported on macOS");
        }

        if target.contains("-none-") || target.contains("nvptx") {
            if build.no_std(*target).is_none() {
                let target = build.config.target_config.entry(target.clone())
                    .or_default();

                target.no_std = true;
            }

            if build.no_std(*target) == Some(false) {
                panic!("All the *-none-* and nvptx* targets are no-std targets")
            }
        }

        // Make sure musl-root is valid
        if target.contains("musl") {
            // If this is a native target (host is also musl) and no musl-root is given,
            // fall back to the system toolchain in /usr before giving up
            if build.musl_root(*target).is_none() && build.config.build == *target {
                let target = build.config.target_config.entry(target.clone())
                    .or_default();
                target.musl_root = Some("/usr".into());
            }
            match build.musl_root(*target) {
                Some(root) => {
                    if fs::metadata(root.join("lib/libc.a")).is_err() {
                        panic!("couldn't find libc.a in musl dir: {}",
                               root.join("lib").display());
                    }
                    if fs::metadata(root.join("lib/libunwind.a")).is_err() {
                        panic!("couldn't find libunwind.a in musl dir: {}",
                               root.join("lib").display());
                    }
                }
                None => {
                    panic!("when targeting MUSL either the rust.musl-root \
                            option or the target.$TARGET.musl-root option must \
                            be specified in config.toml")
                }
            }
        }

        if target.contains("msvc") {
            // There are three builds of cmake on windows: MSVC, MinGW, and
            // Cygwin. The Cygwin build does not have generators for Visual
            // Studio, so detect that here and error.
            let out = output(Command::new("cmake").arg("--help"));
            if !out.contains("Visual Studio") {
                panic!("
cmake does not support Visual Studio generators.

This is likely due to it being an msys/cygwin build of cmake,
rather than the required windows version, built using MinGW
or Visual Studio.

If you are building under msys2 try installing the mingw-w64-x86_64-cmake
package instead of cmake:

$ pacman -R cmake && pacman -S mingw-w64-x86_64-cmake
");
            }
        }
    }

    if let Some(ref s) = build.config.ccache {
        cmd_finder.must_have(s);
    }

    if build.config.channel == "stable" {
        let stage0 = t!(fs::read_to_string(build.src.join("src/stage0.txt")));
        if stage0.contains("\ndev:") {
            panic!("bootstrapping from a dev compiler in a stable release, but \
                    should only be bootstrapping from a released compiler!");
        }
    }
}
