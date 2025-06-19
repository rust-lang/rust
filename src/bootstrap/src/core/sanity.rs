//! Sanity checking performed by bootstrap before actually executing anything.
//!
//! This module contains the implementation of ensuring that the build
//! environment looks reasonable before progressing. This will verify that
//! various programs like git and python exist, along with ensuring that all C
//! compilers for cross-compiling are found.
//!
//! In theory if we get past this phase it's a bug if a build fails, but in
//! practice that's likely not true!

use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::{env, fs};

#[cfg(not(test))]
use crate::builder::Builder;
use crate::builder::Kind;
#[cfg(not(test))]
use crate::core::build_steps::tool;
use crate::core::config::Target;
use crate::utils::exec::command;
use crate::{Build, Subcommand};

pub struct Finder {
    cache: HashMap<OsString, Option<PathBuf>>,
    path: OsString,
}

// During sanity checks, we search for target names to determine if they exist in the compiler's built-in
// target list (`rustc --print target-list`). While a target name may be present in the stage2 compiler,
// it might not yet be included in stage0. In such cases, we handle the targets missing from stage0 in this list.
//
// Targets can be removed from this list once they are present in the stage0 compiler (usually by updating the beta compiler of the bootstrap).
const STAGE0_MISSING_TARGETS: &[&str] = &[
    // just a dummy comment so the list doesn't get onelined
    "loongarch32-unknown-none",
    "loongarch32-unknown-none-softfloat",
];

/// Minimum version threshold for libstdc++ required when using prebuilt LLVM
/// from CI (with`llvm.download-ci-llvm` option).
#[cfg(not(test))]
const LIBSTDCXX_MIN_VERSION_THRESHOLD: usize = 8;

impl Finder {
    pub fn new() -> Self {
        Self { cache: HashMap::new(), path: env::var_os("PATH").unwrap_or_default() }
    }

    pub fn maybe_have<S: Into<OsString>>(&mut self, cmd: S) -> Option<PathBuf> {
        let cmd: OsString = cmd.into();
        let path = &self.path;
        self.cache
            .entry(cmd.clone())
            .or_insert_with(|| {
                for path in env::split_paths(path) {
                    let target = path.join(&cmd);
                    let mut cmd_exe = cmd.clone();
                    cmd_exe.push(".exe");

                    if target.is_file()                   // some/path/git
                    || path.join(&cmd_exe).exists()   // some/path/git.exe
                    || target.join(&cmd_exe).exists()
                    // some/path/git/git.exe
                    {
                        return Some(target);
                    }
                }
                None
            })
            .clone()
    }

    pub fn must_have<S: AsRef<OsStr>>(&mut self, cmd: S) -> PathBuf {
        self.maybe_have(&cmd).unwrap_or_else(|| {
            panic!("\n\ncouldn't find required command: {:?}\n\n", cmd.as_ref());
        })
    }
}

pub fn check(build: &mut Build) {
    let mut skip_target_sanity =
        env::var_os("BOOTSTRAP_SKIP_TARGET_SANITY").is_some_and(|s| s == "1" || s == "true");

    skip_target_sanity |= build.config.cmd.kind() == Kind::Check;

    // Skip target sanity checks when we are doing anything with mir-opt tests or Miri
    let skipped_paths = [OsStr::new("mir-opt"), OsStr::new("miri")];
    skip_target_sanity |= build.config.paths.iter().any(|path| {
        path.components().any(|component| skipped_paths.contains(&component.as_os_str()))
    });

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
    if build.rust_info().is_managed_git_subrepository() {
        cmd_finder.must_have("git");
    }

    // Ensure that a compatible version of libstdc++ is available on the system when using `llvm.download-ci-llvm`.
    #[cfg(not(test))]
    if !build.config.dry_run() && !build.host_target.is_msvc() && build.config.llvm_from_ci {
        let builder = Builder::new(build);
        let libcxx_version = builder.ensure(tool::LibcxxVersionTool { target: build.host_target });

        match libcxx_version {
            tool::LibcxxVersion::Gnu(version) => {
                if LIBSTDCXX_MIN_VERSION_THRESHOLD > version {
                    eprintln!(
                        "\nYour system's libstdc++ version is too old for the `llvm.download-ci-llvm` option."
                    );
                    eprintln!("Current version detected: '{version}'");
                    eprintln!("Minimum required version: '{LIBSTDCXX_MIN_VERSION_THRESHOLD}'");
                    eprintln!(
                        "Consider upgrading libstdc++ or disabling the `llvm.download-ci-llvm` option."
                    );
                    eprintln!(
                        "If you choose to upgrade libstdc++, run `x clean` or delete `build/host/libcxx-version` manually after the upgrade."
                    );
                }
            }
            tool::LibcxxVersion::Llvm(_) => {
                // FIXME: Handle libc++ version check.
            }
        }
    }

    // We need cmake, but only if we're actually building LLVM or sanitizers.
    let building_llvm = !build.config.llvm_from_ci
        && build.hosts.iter().any(|host| {
            build.config.llvm_enabled(*host)
                && build
                    .config
                    .target_config
                    .get(host)
                    .map(|config| config.llvm_config.is_none())
                    .unwrap_or(true)
        });

    let need_cmake = building_llvm || build.config.any_sanitizers_to_build();
    if need_cmake && cmd_finder.maybe_have("cmake").is_none() {
        eprintln!(
            "
Couldn't find required command: cmake

You should install cmake, or set `download-ci-llvm = true` in the
`[llvm]` section of `bootstrap.toml` to download LLVM rather
than building it.
"
        );
        crate::exit!(1);
    }

    build.config.python = build
        .config
        .python
        .take()
        .map(|p| cmd_finder.must_have(p))
        .or_else(|| env::var_os("BOOTSTRAP_PYTHON").map(PathBuf::from)) // set by bootstrap.py
        .or_else(|| cmd_finder.maybe_have("python"))
        .or_else(|| cmd_finder.maybe_have("python3"))
        .or_else(|| cmd_finder.maybe_have("python2"));

    build.config.nodejs = build
        .config
        .nodejs
        .take()
        .map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("node"))
        .or_else(|| cmd_finder.maybe_have("nodejs"));

    build.config.npm = build
        .config
        .npm
        .take()
        .map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("npm"));

    build.config.gdb = build
        .config
        .gdb
        .take()
        .map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("gdb"));

    build.config.reuse = build
        .config
        .reuse
        .take()
        .map(|p| cmd_finder.must_have(p))
        .or_else(|| cmd_finder.maybe_have("reuse"));

    let stage0_supported_target_list: HashSet<String> = command(&build.config.initial_rustc)
        .args(["--print", "target-list"])
        .run_always()
        .run_capture_stdout(&build)
        .stdout()
        .lines()
        .map(|s| s.to_string())
        .collect();

    // Compiler tools like `cc` and `ar` are not configured for cross-targets on certain subcommands
    // because they are not needed.
    //
    // See `cc_detect::find` for more details.
    let skip_tools_checks = build.config.dry_run()
        || matches!(
            build.config.cmd,
            Subcommand::Clean { .. }
                | Subcommand::Check { .. }
                | Subcommand::Suggest { .. }
                | Subcommand::Format { .. }
                | Subcommand::Setup { .. }
        );

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

        // skip check for cross-targets
        if skip_target_sanity && target != &build.host_target {
            continue;
        }

        // Ignore fake targets that are only used for unit tests in bootstrap.
        if cfg!(not(test)) && !skip_target_sanity && !build.local_rebuild {
            let mut has_target = false;
            let target_str = target.to_string();

            let missing_targets_hashset: HashSet<_> =
                STAGE0_MISSING_TARGETS.iter().map(|t| t.to_string()).collect();
            let duplicated_targets: Vec<_> =
                stage0_supported_target_list.intersection(&missing_targets_hashset).collect();

            if !duplicated_targets.is_empty() {
                println!(
                    "Following targets supported from the stage0 compiler, please remove them from STAGE0_MISSING_TARGETS list."
                );
                for duplicated_target in duplicated_targets {
                    println!("  {duplicated_target}");
                }
                std::process::exit(1);
            }

            // Check if it's a built-in target.
            has_target |= stage0_supported_target_list.contains(&target_str);
            has_target |= STAGE0_MISSING_TARGETS.contains(&target_str.as_str());

            if !has_target {
                // This might also be a custom target, so check the target file that could have been specified by the user.
                if target.filepath().is_some_and(|p| p.exists()) {
                    has_target = true;
                } else if let Some(custom_target_path) = env::var_os("RUST_TARGET_PATH") {
                    let mut target_filename = OsString::from(&target_str);
                    // Target filename ends with `.json`.
                    target_filename.push(".json");

                    // Recursively traverse through nested directories.
                    let walker = walkdir::WalkDir::new(custom_target_path).into_iter();
                    for entry in walker.filter_map(|e| e.ok()) {
                        has_target |= entry.file_name() == target_filename;
                    }
                }
            }

            if !has_target {
                panic!(
                    "No such target exists in the target list,\n\
                     make sure to correctly specify the location \
                     of the JSON specification file \
                     for custom targets!\n\
                     Use BOOTSTRAP_SKIP_TARGET_SANITY=1 to \
                     bypass this check."
                );
            }
        }

        if !skip_tools_checks {
            cmd_finder.must_have(build.cc(*target));
            if let Some(ar) = build.ar(*target) {
                cmd_finder.must_have(ar);
            }
        }
    }

    if !skip_tools_checks {
        for host in &build.hosts {
            cmd_finder.must_have(build.cxx(*host).unwrap());

            if build.config.llvm_enabled(*host) {
                // Externally configured LLVM requires FileCheck to exist
                let filecheck = build.llvm_filecheck(build.host_target);
                if !filecheck.starts_with(&build.out)
                    && !filecheck.exists()
                    && build.config.codegen_tests
                {
                    panic!("FileCheck executable {filecheck:?} does not exist");
                }
            }
        }
    }

    for target in &build.targets {
        build
            .config
            .target_config
            .entry(*target)
            .or_insert_with(|| Target::from_triple(&target.triple));

        if (target.contains("-none-") || target.contains("nvptx"))
            && build.no_std(*target) == Some(false)
        {
            panic!("All the *-none-* and nvptx* targets are no-std targets")
        }

        // skip check for cross-targets
        if skip_target_sanity && target != &build.host_target {
            continue;
        }

        // Make sure musl-root is valid.
        if target.contains("musl") && !target.contains("unikraft") {
            // If this is a native target (host is also musl) and no musl-root is given,
            // fall back to the system toolchain in /usr before giving up
            if build.musl_root(*target).is_none() && build.config.is_host_target(*target) {
                let target = build.config.target_config.entry(*target).or_default();
                target.musl_root = Some("/usr".into());
            }
            match build.musl_libdir(*target) {
                Some(libdir) => {
                    if fs::metadata(libdir.join("libc.a")).is_err() {
                        panic!("couldn't find libc.a in musl libdir: {}", libdir.display());
                    }
                }
                None => panic!(
                    "when targeting MUSL either the rust.musl-root \
                            option or the target.$TARGET.musl-root option must \
                            be specified in bootstrap.toml"
                ),
            }
        }

        if need_cmake && target.is_msvc() {
            // There are three builds of cmake on windows: MSVC, MinGW, and
            // Cygwin. The Cygwin build does not have generators for Visual
            // Studio, so detect that here and error.
            let out =
                command("cmake").arg("--help").run_always().run_capture_stdout(&build).stdout();
            if !out.contains("Visual Studio") {
                panic!(
                    "
cmake does not support Visual Studio generators.

This is likely due to it being an msys/cygwin build of cmake,
rather than the required windows version, built using MinGW
or Visual Studio.

If you are building under msys2 try installing the mingw-w64-x86_64-cmake
package instead of cmake:

$ pacman -R cmake && pacman -S mingw-w64-x86_64-cmake
"
                );
            }
        }
    }

    if let Some(ref s) = build.config.ccache {
        cmd_finder.must_have(s);
    }
}
