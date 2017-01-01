// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of rustbuild, the Rust build system.
//!
//! This module, and its descendants, are the implementation of the Rust build
//! system. Most of this build system is backed by Cargo but the outer layer
//! here serves as the ability to orchestrate calling Cargo, sequencing Cargo
//! builds, building artifacts like LLVM, etc. The goals of rustbuild are:
//!
//! * To be an easily understandable, easily extensible, and maintainable build
//!   system.
//! * Leverage standard tools in the Rust ecosystem to build the compiler, aka
//!   crates.io and Cargo.
//! * A standard interface to build across all platforms, including MSVC
//!
//! ## Architecture
//!
//! Although this build system defers most of the complicated logic to Cargo
//! itself, it still needs to maintain a list of targets and dependencies which
//! it can itself perform. Rustbuild is made up of a list of rules with
//! dependencies amongst them (created in the `step` module) and then knows how
//! to execute each in sequence. Each time rustbuild is invoked, it will simply
//! iterate through this list of steps and execute each serially in turn.  For
//! each step rustbuild relies on the step internally being incremental and
//! parallel. Note, though, that the `-j` parameter to rustbuild gets forwarded
//! to appropriate test harnesses and such.
//!
//! Most of the "meaty" steps that matter are backed by Cargo, which does indeed
//! have its own parallelism and incremental management. Later steps, like
//! tests, aren't incremental and simply run the entire suite currently.
//!
//! When you execute `x.py build`, the steps which are executed are:
//!
//! * First, the python script is run. This will automatically download the
//!   stage0 rustc and cargo according to `src/stage0.txt`, or using the cached
//!   versions if they're available. These are then used to compile rustbuild
//!   itself (using Cargo). Finally, control is then transferred to rustbuild.
//!
//! * Rustbuild takes over, performs sanity checks, probes the environment,
//!   reads configuration, builds up a list of steps, and then starts executing
//!   them.
//!
//! * The stage0 libstd is compiled
//! * The stage0 libtest is compiled
//! * The stage0 librustc is compiled
//! * The stage1 compiler is assembled
//! * The stage1 libstd, libtest, librustc are compiled
//! * The stage2 compiler is assembled
//! * The stage2 libstd, libtest, librustc are compiled
//!
//! Each step is driven by a separate Cargo project and rustbuild orchestrates
//! copying files between steps and otherwise preparing for Cargo to run.
//!
//! ## Further information
//!
//! More documentation can be found in each respective module below, and you can
//! also check out the `src/bootstrap/README.md` file for more information.

#![deny(warnings)]

extern crate build_helper;
extern crate cmake;
extern crate filetime;
extern crate gcc;
extern crate getopts;
extern crate num_cpus;
extern crate rustc_serialize;
extern crate toml;

use std::collections::HashMap;
use std::cmp;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Component, PathBuf, Path};
use std::process::Command;

use build_helper::{run_silent, output};

use util::{exe, mtime, libdir, add_lib_path};

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The file/line of the panic
/// * The expression that failed
/// * The error itself
///
/// This is currently used judiciously throughout the build system rather than
/// using a `Result` with `try!`, but this may change one day...
macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

mod cc;
mod channel;
mod check;
mod clean;
mod compile;
mod metadata;
mod config;
mod dist;
mod doc;
mod flags;
mod install;
mod native;
mod sanity;
mod step;
pub mod util;

#[cfg(windows)]
mod job;

#[cfg(not(windows))]
mod job {
    pub unsafe fn setup() {}
}

pub use config::Config;
pub use flags::{Flags, Subcommand};

/// A structure representing a Rust compiler.
///
/// Each compiler has a `stage` that it is associated with and a `host` that
/// corresponds to the platform the compiler runs on. This structure is used as
/// a parameter to many methods below.
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct Compiler<'a> {
    stage: u32,
    host: &'a str,
}

/// Global configuration for the build system.
///
/// This structure transitively contains all configuration for the build system.
/// All filesystem-encoded configuration is in `config`, all flags are in
/// `flags`, and then parsed or probed information is listed in the keys below.
///
/// This structure is a parameter of almost all methods in the build system,
/// although most functions are implemented as free functions rather than
/// methods specifically on this structure itself (to make it easier to
/// organize).
pub struct Build {
    // User-specified configuration via config.toml
    config: Config,

    // User-specified configuration via CLI flags
    flags: Flags,

    // Derived properties from the above two configurations
    cargo: PathBuf,
    rustc: PathBuf,
    src: PathBuf,
    out: PathBuf,
    release: String,
    unstable_features: bool,
    ver_hash: Option<String>,
    short_ver_hash: Option<String>,
    ver_date: Option<String>,
    version: String,
    package_vers: String,
    local_rebuild: bool,

    // Probed tools at runtime
    lldb_version: Option<String>,
    lldb_python_dir: Option<String>,

    // Runtime state filled in later on
    cc: HashMap<String, (gcc::Tool, Option<PathBuf>)>,
    cxx: HashMap<String, gcc::Tool>,
    crates: HashMap<String, Crate>,
    is_sudo: bool,
}

#[derive(Debug)]
struct Crate {
    name: String,
    deps: Vec<String>,
    path: PathBuf,
    doc_step: String,
    build_step: String,
    test_step: String,
    bench_step: String,
}

/// The various "modes" of invoking Cargo.
///
/// These entries currently correspond to the various output directories of the
/// build system, with each mod generating output in a different directory.
#[derive(Clone, Copy)]
pub enum Mode {
    /// This cargo is going to build the standard library, placing output in the
    /// "stageN-std" directory.
    Libstd,

    /// This cargo is going to build libtest, placing output in the
    /// "stageN-test" directory.
    Libtest,

    /// This cargo is going to build librustc and compiler libraries, placing
    /// output in the "stageN-rustc" directory.
    Librustc,

    /// This cargo is going to some build tool, placing output in the
    /// "stageN-tools" directory.
    Tool,
}

impl Build {
    /// Creates a new set of build configuration from the `flags` on the command
    /// line and the filesystem `config`.
    ///
    /// By default all build output will be placed in the current directory.
    pub fn new(flags: Flags, config: Config) -> Build {
        let cwd = t!(env::current_dir());
        let src = flags.src.clone().or_else(|| {
            env::var_os("SRC").map(|x| x.into())
        }).unwrap_or(cwd.clone());
        let out = cwd.join("build");

        let stage0_root = out.join(&config.build).join("stage0/bin");
        let rustc = match config.rustc {
            Some(ref s) => PathBuf::from(s),
            None => stage0_root.join(exe("rustc", &config.build)),
        };
        let cargo = match config.cargo {
            Some(ref s) => PathBuf::from(s),
            None => stage0_root.join(exe("cargo", &config.build)),
        };
        let local_rebuild = config.local_rebuild;

        let is_sudo = match env::var_os("SUDO_USER") {
            Some(sudo_user) => {
                match env::var_os("USER") {
                    Some(user) => user != sudo_user,
                    None => false,
                }
            }
            None => false,
        };

        Build {
            flags: flags,
            config: config,
            cargo: cargo,
            rustc: rustc,
            src: src,
            out: out,

            release: String::new(),
            unstable_features: false,
            ver_hash: None,
            short_ver_hash: None,
            ver_date: None,
            version: String::new(),
            local_rebuild: local_rebuild,
            package_vers: String::new(),
            cc: HashMap::new(),
            cxx: HashMap::new(),
            crates: HashMap::new(),
            lldb_version: None,
            lldb_python_dir: None,
            is_sudo: is_sudo,
        }
    }

    /// Executes the entire build, as configured by the flags and configuration.
    pub fn build(&mut self) {
        unsafe {
            job::setup();
        }

        if let Subcommand::Clean = self.flags.cmd {
            return clean::clean(self);
        }

        self.verbose("finding compilers");
        cc::find(self);
        self.verbose("running sanity check");
        sanity::check(self);
        self.verbose("collecting channel variables");
        channel::collect(self);
        // If local-rust is the same major.minor as the current version, then force a local-rebuild
        let local_version_verbose = output(
            Command::new(&self.rustc).arg("--version").arg("--verbose"));
        let local_release = local_version_verbose
            .lines().filter(|x| x.starts_with("release:"))
            .next().unwrap().trim_left_matches("release:").trim();
        if local_release.split('.').take(2).eq(self.release.split('.').take(2)) {
            self.verbose(&format!("auto-detected local-rebuild {}", local_release));
            self.local_rebuild = true;
        }
        self.verbose("updating submodules");
        self.update_submodules();
        self.verbose("learning about cargo");
        metadata::build(self);

        step::run(self);
    }

    /// Updates all git submodules that we have.
    ///
    /// This will detect if any submodules are out of date an run the necessary
    /// commands to sync them all with upstream.
    fn update_submodules(&self) {
        struct Submodule<'a> {
            path: &'a Path,
            state: State,
        }

        enum State {
            // The submodule may have staged/unstaged changes
            MaybeDirty,
            // Or could be initialized but never updated
            NotInitialized,
            // The submodule, itself, has extra commits but those changes haven't been commited to
            // the (outer) git repository
            OutOfSync,
        }

        if !self.config.submodules {
            return
        }
        if fs::metadata(self.src.join(".git")).is_err() {
            return
        }
        let git = || {
            let mut cmd = Command::new("git");
            cmd.current_dir(&self.src);
            return cmd
        };
        let git_submodule = || {
            let mut cmd = Command::new("git");
            cmd.current_dir(&self.src).arg("submodule");
            return cmd
        };

        // FIXME: this takes a seriously long time to execute on Windows and a
        //        nontrivial amount of time on Unix, we should have a better way
        //        of detecting whether we need to run all the submodule commands
        //        below.
        let out = output(git_submodule().arg("status"));
        let mut submodules = vec![];
        for line in out.lines() {
            // NOTE `git submodule status` output looks like this:
            //
            // -5066b7dcab7e700844b0e2ba71b8af9dc627a59b src/liblibc
            // +b37ef24aa82d2be3a3cc0fe89bf82292f4ca181c src/compiler-rt (remotes/origin/..)
            //  e058ca661692a8d01f8cf9d35939dfe3105ce968 src/jemalloc (3.6.0-533-ge058ca6)
            //
            // The first character can be '-', '+' or ' ' and denotes the `State` of the submodule
            // Right next to this character is the SHA-1 of the submodule HEAD
            // And after that comes the path to the submodule
            let path = Path::new(line[1..].split(' ').skip(1).next().unwrap());
            let state = if line.starts_with('-') {
                State::NotInitialized
            } else if line.starts_with('+') {
                State::OutOfSync
            } else if line.starts_with(' ') {
                State::MaybeDirty
            } else {
                panic!("unexpected git submodule state: {:?}", line.chars().next());
            };

            submodules.push(Submodule { path: path, state: state })
        }

        self.run(git_submodule().arg("sync"));

        for submodule in submodules {
            // If using llvm-root then don't touch the llvm submodule.
            if submodule.path.components().any(|c| c == Component::Normal("llvm".as_ref())) &&
                self.config.target_config.get(&self.config.build)
                    .and_then(|c| c.llvm_config.as_ref()).is_some()
            {
                continue
            }

            if submodule.path.components().any(|c| c == Component::Normal("jemalloc".as_ref())) &&
                !self.config.use_jemalloc
            {
                continue
            }

            // `submodule.path` is the relative path to a submodule (from the repository root)
            // `submodule_path` is the path to a submodule from the cwd

            // use `submodule.path` when e.g. executing a submodule specific command from the
            // repository root
            // use `submodule_path` when e.g. executing a normal git command for the submodule
            // (set via `current_dir`)
            let submodule_path = self.src.join(submodule.path);

            match submodule.state {
                State::MaybeDirty => {
                    // drop staged changes
                    self.run(git().current_dir(&submodule_path)
                                  .args(&["reset", "--hard"]));
                    // drops unstaged changes
                    self.run(git().current_dir(&submodule_path)
                                  .args(&["clean", "-fdx"]));
                },
                State::NotInitialized => {
                    self.run(git_submodule().arg("init").arg(submodule.path));
                    self.run(git_submodule().arg("update").arg(submodule.path));
                },
                State::OutOfSync => {
                    // drops submodule commits that weren't reported to the (outer) git repository
                    self.run(git_submodule().arg("update").arg(submodule.path));
                    self.run(git().current_dir(&submodule_path)
                                  .args(&["reset", "--hard"]));
                    self.run(git().current_dir(&submodule_path)
                                  .args(&["clean", "-fdx"]));
                },
            }
        }
    }

    /// Clear out `dir` if `input` is newer.
    ///
    /// After this executes, it will also ensure that `dir` exists.
    fn clear_if_dirty(&self, dir: &Path, input: &Path) {
        let stamp = dir.join(".stamp");
        if mtime(&stamp) < mtime(input) {
            self.verbose(&format!("Dirty - {}", dir.display()));
            let _ = fs::remove_dir_all(dir);
        } else if stamp.exists() {
            return
        }
        t!(fs::create_dir_all(dir));
        t!(File::create(stamp));
    }

    /// Prepares an invocation of `cargo` to be run.
    ///
    /// This will create a `Command` that represents a pending execution of
    /// Cargo. This cargo will be configured to use `compiler` as the actual
    /// rustc compiler, its output will be scoped by `mode`'s output directory,
    /// it will pass the `--target` flag for the specified `target`, and will be
    /// executing the Cargo command `cmd`.
    fn cargo(&self,
             compiler: &Compiler,
             mode: Mode,
             target: &str,
             cmd: &str) -> Command {
        let mut cargo = Command::new(&self.cargo);
        let out_dir = self.stage_out(compiler, mode);
        cargo.env("CARGO_TARGET_DIR", out_dir)
             .arg(cmd)
             .arg("-j").arg(self.jobs().to_string())
             .arg("--target").arg(target);

        // FIXME: Temporary fix for https://github.com/rust-lang/cargo/issues/3005
        // Force cargo to output binaries with disambiguating hashes in the name
        cargo.env("__CARGO_DEFAULT_LIB_METADATA", "1");

        let stage;
        if compiler.stage == 0 && self.local_rebuild {
            // Assume the local-rebuild rustc already has stage1 features.
            stage = 1;
        } else {
            stage = compiler.stage;
        }

        // Customize the compiler we're running. Specify the compiler to cargo
        // as our shim and then pass it some various options used to configure
        // how the actual compiler itself is called.
        //
        // These variables are primarily all read by
        // src/bootstrap/bin/{rustc.rs,rustdoc.rs}
        cargo.env("RUSTC", self.out.join("bootstrap/debug/rustc"))
             .env("RUSTC_REAL", self.compiler_path(compiler))
             .env("RUSTC_STAGE", stage.to_string())
             .env("RUSTC_DEBUGINFO", self.config.rust_debuginfo.to_string())
             .env("RUSTC_DEBUGINFO_LINES", self.config.rust_debuginfo_lines.to_string())
             .env("RUSTC_CODEGEN_UNITS",
                  self.config.rust_codegen_units.to_string())
             .env("RUSTC_DEBUG_ASSERTIONS",
                  self.config.rust_debug_assertions.to_string())
             .env("RUSTC_SNAPSHOT", &self.rustc)
             .env("RUSTC_SYSROOT", self.sysroot(compiler))
             .env("RUSTC_LIBDIR", self.rustc_libdir(compiler))
             .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_snapshot_libdir())
             .env("RUSTC_RPATH", self.config.rust_rpath.to_string())
             .env("RUSTDOC", self.out.join("bootstrap/debug/rustdoc"))
             .env("RUSTDOC_REAL", self.rustdoc(compiler))
             .env("RUSTC_FLAGS", self.rustc_flags(target).join(" "));

        // Enable usage of unstable features
        cargo.env("RUSTC_BOOTSTRAP", "1");
        self.add_rust_test_threads(&mut cargo);

        // Ignore incremental modes except for stage0, since we're
        // not guaranteeing correctness acros builds if the compiler
        // is changing under your feet.`
        if self.flags.incremental && compiler.stage == 0 {
            let incr_dir = self.incremental_dir(compiler);
            cargo.env("RUSTC_INCREMENTAL", incr_dir);
        }

        let verbose = cmp::max(self.config.verbose, self.flags.verbose);
        cargo.env("RUSTC_VERBOSE", format!("{}", verbose));

        // Specify some various options for build scripts used throughout
        // the build.
        //
        // FIXME: the guard against msvc shouldn't need to be here
        if !target.contains("msvc") {
            cargo.env(format!("CC_{}", target), self.cc(target))
                 .env(format!("AR_{}", target), self.ar(target).unwrap()) // only msvc is None
                 .env(format!("CFLAGS_{}", target), self.cflags(target).join(" "));
        }

        if self.config.channel == "nightly" && compiler.is_final_stage(self) {
            cargo.env("RUSTC_SAVE_ANALYSIS", "api".to_string());
        }

        // Environment variables *required* needed throughout the build
        //
        // FIXME: should update code to not require this env var
        cargo.env("CFG_COMPILER_HOST_TRIPLE", target);

        if self.config.verbose() || self.flags.verbose() {
            cargo.arg("-v");
        }
        // FIXME: cargo bench does not accept `--release`
        if self.config.rust_optimize && cmd != "bench" {
            cargo.arg("--release");
        }
        if self.config.vendor || self.is_sudo {
            cargo.arg("--frozen");
        }
        return cargo
    }

    /// Get a path to the compiler specified.
    fn compiler_path(&self, compiler: &Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.rustc.clone()
        } else {
            self.sysroot(compiler).join("bin").join(exe("rustc", compiler.host))
        }
    }

    /// Get the specified tool built by the specified compiler
    fn tool(&self, compiler: &Compiler, tool: &str) -> PathBuf {
        self.cargo_out(compiler, Mode::Tool, compiler.host)
            .join(exe(tool, compiler.host))
    }

    /// Get the `rustdoc` executable next to the specified compiler
    fn rustdoc(&self, compiler: &Compiler) -> PathBuf {
        let mut rustdoc = self.compiler_path(compiler);
        rustdoc.pop();
        rustdoc.push(exe("rustdoc", compiler.host));
        return rustdoc
    }

    /// Get a `Command` which is ready to run `tool` in `stage` built for
    /// `host`.
    fn tool_cmd(&self, compiler: &Compiler, tool: &str) -> Command {
        let mut cmd = Command::new(self.tool(&compiler, tool));
        self.prepare_tool_cmd(compiler, &mut cmd);
        return cmd
    }

    /// Prepares the `cmd` provided to be able to run the `compiler` provided.
    ///
    /// Notably this munges the dynamic library lookup path to point to the
    /// right location to run `compiler`.
    fn prepare_tool_cmd(&self, compiler: &Compiler, cmd: &mut Command) {
        let host = compiler.host;
        let mut paths = vec![
            self.sysroot_libdir(compiler, compiler.host),
            self.cargo_out(compiler, Mode::Tool, host).join("deps"),
        ];

        // On MSVC a tool may invoke a C compiler (e.g. compiletest in run-make
        // mode) and that C compiler may need some extra PATH modification. Do
        // so here.
        if compiler.host.contains("msvc") {
            let curpaths = env::var_os("PATH").unwrap_or(OsString::new());
            let curpaths = env::split_paths(&curpaths).collect::<Vec<_>>();
            for &(ref k, ref v) in self.cc[compiler.host].0.env() {
                if k != "PATH" {
                    continue
                }
                for path in env::split_paths(v) {
                    if !curpaths.contains(&path) {
                        paths.push(path);
                    }
                }
            }
        }
        add_lib_path(paths, cmd);
    }

    /// Get the space-separated set of activated features for the standard
    /// library.
    fn std_features(&self) -> String {
        let mut features = "panic-unwind".to_string();
        if self.config.debug_jemalloc {
            features.push_str(" debug-jemalloc");
        }
        if self.config.use_jemalloc {
            features.push_str(" jemalloc");
        }
        if self.config.backtrace {
            features.push_str(" backtrace");
        }
        return features
    }

    /// Get the space-separated set of activated features for the compiler.
    fn rustc_features(&self) -> String {
        let mut features = String::new();
        if self.config.use_jemalloc {
            features.push_str(" jemalloc");
        }
        return features
    }

    /// Component directory that Cargo will produce output into (e.g.
    /// release/debug)
    fn cargo_dir(&self) -> &'static str {
        if self.config.rust_optimize {"release"} else {"debug"}
    }

    /// Returns the sysroot for the `compiler` specified that *this build system
    /// generates*.
    ///
    /// That is, the sysroot for the stage0 compiler is not what the compiler
    /// thinks it is by default, but it's the same as the default for stages
    /// 1-3.
    fn sysroot(&self, compiler: &Compiler) -> PathBuf {
        if compiler.stage == 0 {
            self.out.join(compiler.host).join("stage0-sysroot")
        } else {
            self.out.join(compiler.host).join(format!("stage{}", compiler.stage))
        }
    }

    /// Get the directory for incremental by-products when using the
    /// given compiler.
    fn incremental_dir(&self, compiler: &Compiler) -> PathBuf {
        self.out.join(compiler.host).join(format!("stage{}-incremental", compiler.stage))
    }

    /// Returns the libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
    fn sysroot_libdir(&self, compiler: &Compiler, target: &str) -> PathBuf {
        self.sysroot(compiler).join("lib").join("rustlib")
            .join(target).join("lib")
    }

    /// Returns the root directory for all output generated in a particular
    /// stage when running with a particular host compiler.
    ///
    /// The mode indicates what the root directory is for.
    fn stage_out(&self, compiler: &Compiler, mode: Mode) -> PathBuf {
        let suffix = match mode {
            Mode::Libstd => "-std",
            Mode::Libtest => "-test",
            Mode::Tool => "-tools",
            Mode::Librustc => "-rustc",
        };
        self.out.join(compiler.host)
                .join(format!("stage{}{}", compiler.stage, suffix))
    }

    /// Returns the root output directory for all Cargo output in a given stage,
    /// running a particular comipler, wehther or not we're building the
    /// standard library, and targeting the specified architecture.
    fn cargo_out(&self,
                 compiler: &Compiler,
                 mode: Mode,
                 target: &str) -> PathBuf {
        self.stage_out(compiler, mode).join(target).join(self.cargo_dir())
    }

    /// Root output directory for LLVM compiled for `target`
    ///
    /// Note that if LLVM is configured externally then the directory returned
    /// will likely be empty.
    fn llvm_out(&self, target: &str) -> PathBuf {
        self.out.join(target).join("llvm")
    }

    /// Output directory for all documentation for a target
    fn doc_out(&self, target: &str) -> PathBuf {
        self.out.join(target).join("doc")
    }

    /// Returns true if no custom `llvm-config` is set for the specified target.
    ///
    /// If no custom `llvm-config` was specified then Rust's llvm will be used.
    fn is_rust_llvm(&self, target: &str) -> bool {
        match self.config.target_config.get(target) {
            Some(ref c) => c.llvm_config.is_none(),
            None => true
        }
    }

    /// Returns the path to `llvm-config` for the specified target.
    ///
    /// If a custom `llvm-config` was specified for target then that's returned
    /// instead.
    fn llvm_config(&self, target: &str) -> PathBuf {
        let target_config = self.config.target_config.get(target);
        if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            s.clone()
        } else {
            self.llvm_out(&self.config.build).join("bin")
                .join(exe("llvm-config", target))
        }
    }

    /// Returns the path to `FileCheck` binary for the specified target
    fn llvm_filecheck(&self, target: &str) -> PathBuf {
        let target_config = self.config.target_config.get(target);
        if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            let llvm_bindir = output(Command::new(s).arg("--bindir"));
            Path::new(llvm_bindir.trim()).join(exe("FileCheck", target))
        } else {
            let base = self.llvm_out(&self.config.build).join("build");
            let exe = exe("FileCheck", target);
            if self.config.build.contains("msvc") {
                base.join("Release/bin").join(exe)
            } else {
                base.join("bin").join(exe)
            }
        }
    }

    /// Root output directory for rust_test_helpers library compiled for
    /// `target`
    fn test_helpers_out(&self, target: &str) -> PathBuf {
        self.out.join(target).join("rust-test-helpers")
    }

    /// Adds the compiler's directory of dynamic libraries to `cmd`'s dynamic
    /// library lookup path.
    fn add_rustc_lib_path(&self, compiler: &Compiler, cmd: &mut Command) {
        // Windows doesn't need dylib path munging because the dlls for the
        // compiler live next to the compiler and the system will find them
        // automatically.
        if cfg!(windows) {
            return
        }

        add_lib_path(vec![self.rustc_libdir(compiler)], cmd);
    }

    /// Adds the `RUST_TEST_THREADS` env var if necessary
    fn add_rust_test_threads(&self, cmd: &mut Command) {
        if env::var_os("RUST_TEST_THREADS").is_none() {
            cmd.env("RUST_TEST_THREADS", self.jobs().to_string());
        }
    }

    /// Returns the compiler's libdir where it stores the dynamic libraries that
    /// it itself links against.
    ///
    /// For example this returns `<sysroot>/lib` on Unix and `<sysroot>/bin` on
    /// Windows.
    fn rustc_libdir(&self, compiler: &Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.rustc_snapshot_libdir()
        } else {
            self.sysroot(compiler).join(libdir(compiler.host))
        }
    }

    /// Returns the libdir of the snapshot compiler.
    fn rustc_snapshot_libdir(&self) -> PathBuf {
        self.rustc.parent().unwrap().parent().unwrap()
            .join(libdir(&self.config.build))
    }

    /// Runs a command, printing out nice contextual information if it fails.
    fn run(&self, cmd: &mut Command) {
        self.verbose(&format!("running: {:?}", cmd));
        run_silent(cmd)
    }

    /// Prints a message if this build is configured in verbose mode.
    fn verbose(&self, msg: &str) {
        if self.flags.verbose() || self.config.verbose() {
            println!("{}", msg);
        }
    }

    /// Returns the number of parallel jobs that have been configured for this
    /// build.
    fn jobs(&self) -> u32 {
        self.flags.jobs.unwrap_or(num_cpus::get() as u32)
    }

    /// Returns the path to the C compiler for the target specified.
    fn cc(&self, target: &str) -> &Path {
        self.cc[target].0.path()
    }

    /// Returns a list of flags to pass to the C compiler for the target
    /// specified.
    fn cflags(&self, target: &str) -> Vec<String> {
        // Filter out -O and /O (the optimization flags) that we picked up from
        // gcc-rs because the build scripts will determine that for themselves.
        let mut base = self.cc[target].0.args().iter()
                           .map(|s| s.to_string_lossy().into_owned())
                           .filter(|s| !s.starts_with("-O") && !s.starts_with("/O"))
                           .collect::<Vec<_>>();

        // If we're compiling on OSX then we add a few unconditional flags
        // indicating that we want libc++ (more filled out than libstdc++) and
        // we want to compile for 10.7. This way we can ensure that
        // LLVM/jemalloc/etc are all properly compiled.
        if target.contains("apple-darwin") {
            base.push("-stdlib=libc++".into());
        }
        // This is a hack, because newer binutils broke things on some vms/distros
        // (i.e., linking against unknown relocs disabled by the following flag)
        // See: https://github.com/rust-lang/rust/issues/34978
        match target {
            "i586-unknown-linux-gnu" |
            "i686-unknown-linux-musl" |
            "x86_64-unknown-linux-musl" => {
                base.push("-Wa,-mrelax-relocations=no".into());
            },
            _ => {},
        }
        return base
    }

    /// Returns the path to the `ar` archive utility for the target specified.
    fn ar(&self, target: &str) -> Option<&Path> {
        self.cc[target].1.as_ref().map(|p| &**p)
    }

    /// Returns the path to the C++ compiler for the target specified, may panic
    /// if no C++ compiler was configured for the target.
    fn cxx(&self, target: &str) -> &Path {
        match self.cxx.get(target) {
            Some(p) => p.path(),
            None => panic!("\n\ntarget `{}` is not configured as a host,
                            only as a target\n\n", target),
        }
    }

    /// Returns flags to pass to the compiler to generate code for `target`.
    fn rustc_flags(&self, target: &str) -> Vec<String> {
        // New flags should be added here with great caution!
        //
        // It's quite unfortunate to **require** flags to generate code for a
        // target, so it should only be passed here if absolutely necessary!
        // Most default configuration should be done through target specs rather
        // than an entry here.

        let mut base = Vec::new();
        if target != self.config.build && !target.contains("msvc") &&
            !target.contains("emscripten") {
            base.push(format!("-Clinker={}", self.cc(target).display()));
        }
        return base
    }

    /// Returns the "musl root" for this `target`, if defined
    fn musl_root(&self, target: &str) -> Option<&Path> {
        self.config.target_config.get(target)
            .and_then(|t| t.musl_root.as_ref())
            .or(self.config.musl_root.as_ref())
            .map(|p| &**p)
    }

    /// Path to the python interpreter to use
    fn python(&self) -> &Path {
        self.config.python.as_ref().unwrap()
    }

    /// Tests whether the `compiler` compiling for `target` should be forced to
    /// use a stage1 compiler instead.
    ///
    /// Currently, by default, the build system does not perform a "full
    /// bootstrap" by default where we compile the compiler three times.
    /// Instead, we compile the compiler two times. The final stage (stage2)
    /// just copies the libraries from the previous stage, which is what this
    /// method detects.
    ///
    /// Here we return `true` if:
    ///
    /// * The build isn't performing a full bootstrap
    /// * The `compiler` is in the final stage, 2
    /// * We're not cross-compiling, so the artifacts are already available in
    ///   stage1
    ///
    /// When all of these conditions are met the build will lift artifacts from
    /// the previous stage forward.
    fn force_use_stage1(&self, compiler: &Compiler, target: &str) -> bool {
        !self.config.full_bootstrap &&
            compiler.stage >= 2 &&
            self.config.host.iter().any(|h| h == target)
    }
}

impl<'a> Compiler<'a> {
    /// Creates a new complier for the specified stage/host
    fn new(stage: u32, host: &'a str) -> Compiler<'a> {
        Compiler { stage: stage, host: host }
    }

    /// Returns whether this is a snapshot compiler for `build`'s configuration
    fn is_snapshot(&self, build: &Build) -> bool {
        self.stage == 0 && self.host == build.config.build
    }

    /// Returns if this compiler should be treated as a final stage one in the
    /// current build session.
    /// This takes into account whether we're performing a full bootstrap or
    /// not; don't directly compare the stage with `2`!
    fn is_final_stage(&self, build: &Build) -> bool {
        let final_stage = if build.config.full_bootstrap { 2 } else { 1 };
        self.stage >= final_stage
    }
}
