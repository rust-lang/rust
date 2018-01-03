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
//! The build system defers most of the complicated logic managing invocations
//! of rustc and rustdoc to Cargo itself. However, moving through various stages
//! and copying artifacts is still necessary for it to do. Each time rustbuild
//! is invoked, it will iterate through the list of predefined steps and execute
//! each serially in turn if it matches the paths passed or is a default rule.
//! For each step rustbuild relies on the step internally being incremental and
//! parallel. Note, though, that the `-j` parameter to rustbuild gets forwarded
//! to appropriate test harnesses and such.
//!
//! Most of the "meaty" steps that matter are backed by Cargo, which does indeed
//! have its own parallelism and incremental management. Later steps, like
//! tests, aren't incremental and simply run the entire suite currently.
//! However, compiletest itself tries to avoid running tests when the artifacts
//! that are involved (mainly the compiler) haven't changed.
//!
//! When you execute `x.py build`, the steps which are executed are:
//!
//! * First, the python script is run. This will automatically download the
//!   stage0 rustc and cargo according to `src/stage0.txt`, or use the cached
//!   versions if they're available. These are then used to compile rustbuild
//!   itself (using Cargo). Finally, control is then transferred to rustbuild.
//!
//! * Rustbuild takes over, performs sanity checks, probes the environment,
//!   reads configuration, and starts executing steps as it reads the command
//!   line arguments (paths) or going through the default rules.
//!
//!   The build output will be something like the following:
//!
//!   Building stage0 std artifacts
//!   Copying stage0 std
//!   Building stage0 test artifacts
//!   Copying stage0 test
//!   Building stage0 compiler artifacts
//!   Copying stage0 rustc
//!   Assembling stage1 compiler
//!   Building stage1 std artifacts
//!   Copying stage1 std
//!   Building stage1 test artifacts
//!   Copying stage1 test
//!   Building stage1 compiler artifacts
//!   Copying stage1 rustc
//!   Assembling stage2 compiler
//!   Uplifting stage1 std
//!   Uplifting stage1 test
//!   Uplifting stage1 rustc
//!
//! Let's disect that a little:
//!
//! ## Building stage0 {std,test,compiler} artifacts
//!
//! These steps use the provided (downloaded, usually) compiler to compile the
//! local Rust source into libraries we can use.
//!
//! ## Copying stage0 {std,test,rustc}
//!
//! This copies the build output from Cargo into
//! `build/$HOST/stage0-sysroot/lib/rustlib/$ARCH/lib`. FIXME: This step's
//! documentation should be expanded -- the information already here may be
//! incorrect.
//!
//! ## Assembling stage1 compiler
//!
//! This copies the libraries we built in "building stage0 ... artifacts" into
//! the stage1 compiler's lib directory. These are the host libraries that the
//! compiler itself uses to run. These aren't actually used by artifacts the new
//! compiler generates. This step also copies the rustc and rustdoc binaries we
//! generated into build/$HOST/stage/bin.
//!
//! The stage1/bin/rustc is a fully functional compiler, but it doesn't yet have
//! any libraries to link built binaries or libraries to. The next 3 steps will
//! provide those libraries for it; they are mostly equivalent to constructing
//! the stage1/bin compiler so we don't go through them individually.
//!
//! ## Uplifting stage1 {std,test,rustc}
//!
//! This step copies the libraries from the stage1 compiler sysroot into the
//! stage2 compiler. This is done to avoid rebuilding the compiler; libraries
//! we'd build in this step should be identical (in function, if not necessarily
//! identical on disk) so there's no need to recompile the compiler again. Note
//! that if you want to, you can enable the full-bootstrap option to change this
//! behavior.
//!
//! Each step is driven by a separate Cargo project and rustbuild orchestrates
//! copying files between steps and otherwise preparing for Cargo to run.
//!
//! ## Further information
//!
//! More documentation can be found in each respective module below, and you can
//! also check out the `src/bootstrap/README.md` file for more information.

#![deny(warnings)]
#![allow(stable_features)]
#![feature(associated_consts)]

#[macro_use]
extern crate build_helper;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate lazy_static;
extern crate serde_json;
extern crate cmake;
extern crate filetime;
extern crate cc;
extern crate getopts;
extern crate num_cpus;
extern crate toml;

#[cfg(unix)]
extern crate libc;

use std::cell::RefCell;
use std::collections::{HashSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::{PathBuf, Path};
use std::process::{self, Command};
use std::slice;

use build_helper::{run_silent, run_suppressed, try_run_silent, try_run_suppressed, output, mtime};

use util::{exe, libdir, OutputFolder, CiEnv};

mod cc_detect;
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
pub mod util;
mod builder;
mod cache;
mod tool;
mod toolstate;

#[cfg(windows)]
mod job;

#[cfg(unix)]
mod job {
    use libc;

    pub unsafe fn setup(build: &mut ::Build) {
        if build.config.low_priority {
            libc::setpriority(libc::PRIO_PGRP as _, 0, 10);
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod job {
    pub unsafe fn setup(_build: &mut ::Build) {
    }
}

pub use config::Config;
use flags::Subcommand;
use cache::{Interned, INTERNER};
use toolstate::ToolState;

/// A structure representing a Rust compiler.
///
/// Each compiler has a `stage` that it is associated with and a `host` that
/// corresponds to the platform the compiler runs on. This structure is used as
/// a parameter to many methods below.
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct Compiler {
    stage: u32,
    host: Interned<String>,
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

    // Derived properties from the above two configurations
    src: PathBuf,
    out: PathBuf,
    rust_info: channel::GitInfo,
    cargo_info: channel::GitInfo,
    rls_info: channel::GitInfo,
    rustfmt_info: channel::GitInfo,
    local_rebuild: bool,
    fail_fast: bool,
    verbosity: usize,

    // Targets for which to build.
    build: Interned<String>,
    hosts: Vec<Interned<String>>,
    targets: Vec<Interned<String>>,

    // Stage 0 (downloaded) compiler and cargo or their local rust equivalents.
    initial_rustc: PathBuf,
    initial_cargo: PathBuf,

    // Probed tools at runtime
    lldb_version: Option<String>,
    lldb_python_dir: Option<String>,

    // Runtime state filled in later on
    // C/C++ compilers and archiver for all targets
    cc: HashMap<Interned<String>, cc::Tool>,
    cxx: HashMap<Interned<String>, cc::Tool>,
    ar: HashMap<Interned<String>, PathBuf>,
    // Misc
    crates: HashMap<Interned<String>, Crate>,
    is_sudo: bool,
    ci_env: CiEnv,
    delayed_failures: RefCell<Vec<String>>,
}

#[derive(Debug)]
struct Crate {
    name: Interned<String>,
    version: String,
    deps: Vec<Interned<String>>,
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
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Build the standard library, placing output in the "stageN-std" directory.
    Libstd,

    /// Build libtest, placing output in the "stageN-test" directory.
    Libtest,

    /// Build librustc and compiler libraries, placing output in the "stageN-rustc" directory.
    Librustc,

    /// Build some tool, placing output in the "stageN-tools" directory.
    Tool,
}

impl Build {
    /// Creates a new set of build configuration from the `flags` on the command
    /// line and the filesystem `config`.
    ///
    /// By default all build output will be placed in the current directory.
    pub fn new(config: Config) -> Build {
        let cwd = t!(env::current_dir());
        let src = config.src.clone();
        let out = cwd.join("build");

        let is_sudo = match env::var_os("SUDO_USER") {
            Some(sudo_user) => {
                match env::var_os("USER") {
                    Some(user) => user != sudo_user,
                    None => false,
                }
            }
            None => false,
        };
        let rust_info = channel::GitInfo::new(&config, &src);
        let cargo_info = channel::GitInfo::new(&config, &src.join("src/tools/cargo"));
        let rls_info = channel::GitInfo::new(&config, &src.join("src/tools/rls"));
        let rustfmt_info = channel::GitInfo::new(&config, &src.join("src/tools/rustfmt"));

        Build {
            initial_rustc: config.initial_rustc.clone(),
            initial_cargo: config.initial_cargo.clone(),
            local_rebuild: config.local_rebuild,
            fail_fast: config.cmd.fail_fast(),
            verbosity: config.verbose,

            build: config.build,
            hosts: config.hosts.clone(),
            targets: config.targets.clone(),

            config,
            src,
            out,

            rust_info,
            cargo_info,
            rls_info,
            rustfmt_info,
            cc: HashMap::new(),
            cxx: HashMap::new(),
            ar: HashMap::new(),
            crates: HashMap::new(),
            lldb_version: None,
            lldb_python_dir: None,
            is_sudo,
            ci_env: CiEnv::current(),
            delayed_failures: RefCell::new(Vec::new()),
        }
    }

    pub fn build_triple(&self) -> &[Interned<String>] {
        unsafe {
            slice::from_raw_parts(&self.build, 1)
        }
    }

    /// Executes the entire build, as configured by the flags and configuration.
    pub fn build(&mut self) {
        unsafe {
            job::setup(self);
        }

        if let Subcommand::Clean { all } = self.config.cmd {
            return clean::clean(self, all);
        }

        self.verbose("finding compilers");
        cc_detect::find(self);
        self.verbose("running sanity check");
        sanity::check(self);
        // If local-rust is the same major.minor as the current version, then force a local-rebuild
        let local_version_verbose = output(
            Command::new(&self.initial_rustc).arg("--version").arg("--verbose"));
        let local_release = local_version_verbose
            .lines().filter(|x| x.starts_with("release:"))
            .next().unwrap().trim_left_matches("release:").trim();
        let my_version = channel::CFG_RELEASE_NUM;
        if local_release.split('.').take(2).eq(my_version.split('.').take(2)) {
            self.verbose(&format!("auto-detected local-rebuild {}", local_release));
            self.local_rebuild = true;
        }
        self.verbose("learning about cargo");
        metadata::build(self);

        builder::Builder::run(&self);

        // Check for postponed failures from `test --no-fail-fast`.
        let failures = self.delayed_failures.borrow();
        if failures.len() > 0 {
            println!("\n{} command(s) did not execute successfully:\n", failures.len());
            for failure in failures.iter() {
                println!("  - {}\n", failure);
            }
            process::exit(1);
        }
    }

    /// Clear out `dir` if `input` is newer.
    ///
    /// After this executes, it will also ensure that `dir` exists.
    fn clear_if_dirty(&self, dir: &Path, input: &Path) -> bool {
        let stamp = dir.join(".stamp");
        let mut cleared = false;
        if mtime(&stamp) < mtime(input) {
            self.verbose(&format!("Dirty - {}", dir.display()));
            let _ = fs::remove_dir_all(dir);
            cleared = true;
        } else if stamp.exists() {
            return cleared;
        }
        t!(fs::create_dir_all(dir));
        t!(File::create(stamp));
        cleared
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
        if self.config.profiler {
            features.push_str(" profiler");
        }
        features
    }

    /// Get the space-separated set of activated features for the compiler.
    fn rustc_features(&self) -> String {
        let mut features = String::new();
        if self.config.use_jemalloc {
            features.push_str(" jemalloc");
        }
        if self.config.llvm_enabled {
            features.push_str(" llvm");
        }
        features
    }

    /// Component directory that Cargo will produce output into (e.g.
    /// release/debug)
    fn cargo_dir(&self) -> &'static str {
        if self.config.rust_optimize {"release"} else {"debug"}
    }

    fn tools_dir(&self, compiler: Compiler) -> PathBuf {
        let out = self.out.join(&*compiler.host).join(format!("stage{}-tools-bin", compiler.stage));
        t!(fs::create_dir_all(&out));
        out
    }

    /// Get the directory for incremental by-products when using the
    /// given compiler.
    fn incremental_dir(&self, compiler: Compiler) -> PathBuf {
        self.out.join(&*compiler.host).join(format!("stage{}-incremental", compiler.stage))
    }

    /// Returns the root directory for all output generated in a particular
    /// stage when running with a particular host compiler.
    ///
    /// The mode indicates what the root directory is for.
    fn stage_out(&self, compiler: Compiler, mode: Mode) -> PathBuf {
        let suffix = match mode {
            Mode::Libstd => "-std",
            Mode::Libtest => "-test",
            Mode::Tool => "-tools",
            Mode::Librustc => "-rustc",
        };
        self.out.join(&*compiler.host)
                .join(format!("stage{}{}", compiler.stage, suffix))
    }

    /// Returns the root output directory for all Cargo output in a given stage,
    /// running a particular compiler, whether or not we're building the
    /// standard library, and targeting the specified architecture.
    fn cargo_out(&self,
                 compiler: Compiler,
                 mode: Mode,
                 target: Interned<String>) -> PathBuf {
        self.stage_out(compiler, mode).join(&*target).join(self.cargo_dir())
    }

    /// Root output directory for LLVM compiled for `target`
    ///
    /// Note that if LLVM is configured externally then the directory returned
    /// will likely be empty.
    fn llvm_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("llvm")
    }

    /// Output directory for all documentation for a target
    fn doc_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("doc")
    }

    /// Output directory for some generated md crate documentation for a target (temporary)
    fn md_doc_out(&self, target: Interned<String>) -> Interned<PathBuf> {
        INTERNER.intern_path(self.out.join(&*target).join("md-doc"))
    }

    /// Output directory for all crate documentation for a target (temporary)
    ///
    /// The artifacts here are then copied into `doc_out` above.
    fn crate_doc_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("crate-docs")
    }

    /// Returns true if no custom `llvm-config` is set for the specified target.
    ///
    /// If no custom `llvm-config` was specified then Rust's llvm will be used.
    fn is_rust_llvm(&self, target: Interned<String>) -> bool {
        match self.config.target_config.get(&target) {
            Some(ref c) => c.llvm_config.is_none(),
            None => true
        }
    }

    /// Returns the path to `llvm-config` for the specified target.
    ///
    /// If a custom `llvm-config` was specified for target then that's returned
    /// instead.
    fn llvm_config(&self, target: Interned<String>) -> PathBuf {
        let target_config = self.config.target_config.get(&target);
        if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            s.clone()
        } else {
            self.llvm_out(self.config.build).join("bin")
                .join(exe("llvm-config", &*target))
        }
    }

    /// Returns the path to `FileCheck` binary for the specified target
    fn llvm_filecheck(&self, target: Interned<String>) -> PathBuf {
        let target_config = self.config.target_config.get(&target);
        if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            let llvm_bindir = output(Command::new(s).arg("--bindir"));
            Path::new(llvm_bindir.trim()).join(exe("FileCheck", &*target))
        } else {
            let base = self.llvm_out(self.config.build).join("build");
            let exe = exe("FileCheck", &*target);
            if !self.config.ninja && self.config.build.contains("msvc") {
                base.join("Release/bin").join(exe)
            } else {
                base.join("bin").join(exe)
            }
        }
    }

    /// Directory for libraries built from C/C++ code and shared between stages.
    fn native_dir(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("native")
    }

    /// Root output directory for rust_test_helpers library compiled for
    /// `target`
    fn test_helpers_out(&self, target: Interned<String>) -> PathBuf {
        self.native_dir(target).join("rust-test-helpers")
    }

    /// Adds the `RUST_TEST_THREADS` env var if necessary
    fn add_rust_test_threads(&self, cmd: &mut Command) {
        if env::var_os("RUST_TEST_THREADS").is_none() {
            cmd.env("RUST_TEST_THREADS", self.jobs().to_string());
        }
    }

    /// Returns the libdir of the snapshot compiler.
    fn rustc_snapshot_libdir(&self) -> PathBuf {
        self.initial_rustc.parent().unwrap().parent().unwrap()
            .join(libdir(&self.config.build))
    }

    /// Runs a command, printing out nice contextual information if it fails.
    fn run(&self, cmd: &mut Command) {
        self.verbose(&format!("running: {:?}", cmd));
        run_silent(cmd)
    }

    /// Runs a command, printing out nice contextual information if it fails.
    fn run_quiet(&self, cmd: &mut Command) {
        self.verbose(&format!("running: {:?}", cmd));
        run_suppressed(cmd)
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    fn try_run(&self, cmd: &mut Command) -> bool {
        self.verbose(&format!("running: {:?}", cmd));
        try_run_silent(cmd)
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    fn try_run_quiet(&self, cmd: &mut Command) -> bool {
        self.verbose(&format!("running: {:?}", cmd));
        try_run_suppressed(cmd)
    }

    pub fn is_verbose(&self) -> bool {
        self.verbosity > 0
    }

    pub fn is_very_verbose(&self) -> bool {
        self.verbosity > 1
    }

    /// Prints a message if this build is configured in verbose mode.
    fn verbose(&self, msg: &str) {
        if self.is_verbose() {
            println!("{}", msg);
        }
    }

    /// Returns the number of parallel jobs that have been configured for this
    /// build.
    fn jobs(&self) -> u32 {
        self.config.jobs.unwrap_or_else(|| num_cpus::get() as u32)
    }

    /// Returns the path to the C compiler for the target specified.
    fn cc(&self, target: Interned<String>) -> &Path {
        self.cc[&target].path()
    }

    /// Returns a list of flags to pass to the C compiler for the target
    /// specified.
    fn cflags(&self, target: Interned<String>) -> Vec<String> {
        // Filter out -O and /O (the optimization flags) that we picked up from
        // cc-rs because the build scripts will determine that for themselves.
        let mut base = self.cc[&target].args().iter()
                           .map(|s| s.to_string_lossy().into_owned())
                           .filter(|s| !s.starts_with("-O") && !s.starts_with("/O"))
                           .collect::<Vec<_>>();

        // If we're compiling on macOS then we add a few unconditional flags
        // indicating that we want libc++ (more filled out than libstdc++) and
        // we want to compile for 10.7. This way we can ensure that
        // LLVM/jemalloc/etc are all properly compiled.
        if target.contains("apple-darwin") {
            base.push("-stdlib=libc++".into());
        }

        // Work around an apparently bad MinGW / GCC optimization,
        // See: http://lists.llvm.org/pipermail/cfe-dev/2016-December/051980.html
        // See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=78936
        if &*target == "i686-pc-windows-gnu" {
            base.push("-fno-omit-frame-pointer".into());
        }
        base
    }

    /// Returns the path to the `ar` archive utility for the target specified.
    fn ar(&self, target: Interned<String>) -> Option<&Path> {
        self.ar.get(&target).map(|p| &**p)
    }

    /// Returns the path to the C++ compiler for the target specified.
    fn cxx(&self, target: Interned<String>) -> Result<&Path, String> {
        match self.cxx.get(&target) {
            Some(p) => Ok(p.path()),
            None => Err(format!(
                    "target `{}` is not configured as a host, only as a target",
                    target))
        }
    }

    /// Returns the path to the linker for the given target if it needs to be overriden.
    fn linker(&self, target: Interned<String>) -> Option<&Path> {
        if let Some(linker) = self.config.target_config.get(&target)
                                                       .and_then(|c| c.linker.as_ref()) {
            Some(linker)
        } else if target != self.config.build &&
                  !target.contains("msvc") && !target.contains("emscripten") {
            Some(self.cc(target))
        } else {
            None
        }
    }

    /// Returns if this target should statically link the C runtime, if specified
    fn crt_static(&self, target: Interned<String>) -> Option<bool> {
        if target.contains("pc-windows-msvc") {
            Some(true)
        } else {
            self.config.target_config.get(&target)
                .and_then(|t| t.crt_static)
        }
    }

    /// Returns the "musl root" for this `target`, if defined
    fn musl_root(&self, target: Interned<String>) -> Option<&Path> {
        self.config.target_config.get(&target)
            .and_then(|t| t.musl_root.as_ref())
            .or(self.config.musl_root.as_ref())
            .map(|p| &**p)
    }

    /// Returns whether the target will be tested using the `remote-test-client`
    /// and `remote-test-server` binaries.
    fn remote_tested(&self, target: Interned<String>) -> bool {
        self.qemu_rootfs(target).is_some() || target.contains("android") ||
        env::var_os("TEST_DEVICE_ADDR").is_some()
    }

    /// Returns the root of the "rootfs" image that this target will be using,
    /// if one was configured.
    ///
    /// If `Some` is returned then that means that tests for this target are
    /// emulated with QEMU and binaries will need to be shipped to the emulator.
    fn qemu_rootfs(&self, target: Interned<String>) -> Option<&Path> {
        self.config.target_config.get(&target)
            .and_then(|t| t.qemu_rootfs.as_ref())
            .map(|p| &**p)
    }

    /// Path to the python interpreter to use
    fn python(&self) -> &Path {
        self.config.python.as_ref().unwrap()
    }

    /// Temporary directory that extended error information is emitted to.
    fn extended_error_dir(&self) -> PathBuf {
        self.out.join("tmp/extended-error-metadata")
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
    fn force_use_stage1(&self, compiler: Compiler, target: Interned<String>) -> bool {
        !self.config.full_bootstrap &&
            compiler.stage >= 2 &&
            (self.hosts.iter().any(|h| *h == target) || target == self.build)
    }

    /// Returns the directory that OpenSSL artifacts are compiled into if
    /// configured to do so.
    fn openssl_dir(&self, target: Interned<String>) -> Option<PathBuf> {
        // OpenSSL not used on Windows
        if target.contains("windows") {
            None
        } else if self.config.openssl_static {
            Some(self.out.join(&*target).join("openssl"))
        } else {
            None
        }
    }

    /// Returns the directory that OpenSSL artifacts are installed into if
    /// configured as such.
    fn openssl_install_dir(&self, target: Interned<String>) -> Option<PathBuf> {
        self.openssl_dir(target).map(|p| p.join("install"))
    }

    /// Given `num` in the form "a.b.c" return a "release string" which
    /// describes the release version number.
    ///
    /// For example on nightly this returns "a.b.c-nightly", on beta it returns
    /// "a.b.c-beta.1" and on stable it just returns "a.b.c".
    fn release(&self, num: &str) -> String {
        match &self.config.channel[..] {
            "stable" => num.to_string(),
            "beta" => format!("{}-beta{}", num, channel::CFG_PRERELEASE_VERSION),
            "nightly" => format!("{}-nightly", num),
            _ => format!("{}-dev", num),
        }
    }

    /// Returns the value of `release` above for Rust itself.
    fn rust_release(&self) -> String {
        self.release(channel::CFG_RELEASE_NUM)
    }

    /// Returns the "package version" for a component given the `num` release
    /// number.
    ///
    /// The package version is typically what shows up in the names of tarballs.
    /// For channels like beta/nightly it's just the channel name, otherwise
    /// it's the `num` provided.
    fn package_vers(&self, num: &str) -> String {
        match &self.config.channel[..] {
            "stable" => num.to_string(),
            "beta" => "beta".to_string(),
            "nightly" => "nightly".to_string(),
            _ => format!("{}-dev", num),
        }
    }

    /// Returns the value of `package_vers` above for Rust itself.
    fn rust_package_vers(&self) -> String {
        self.package_vers(channel::CFG_RELEASE_NUM)
    }

    /// Returns the value of `package_vers` above for Cargo
    fn cargo_package_vers(&self) -> String {
        self.package_vers(&self.release_num("cargo"))
    }

    /// Returns the value of `package_vers` above for rls
    fn rls_package_vers(&self) -> String {
        self.package_vers(&self.release_num("rls"))
    }

    /// Returns the value of `package_vers` above for rustfmt
    fn rustfmt_package_vers(&self) -> String {
        self.package_vers(&self.release_num("rustfmt"))
    }

    /// Returns the `version` string associated with this compiler for Rust
    /// itself.
    ///
    /// Note that this is a descriptive string which includes the commit date,
    /// sha, version, etc.
    fn rust_version(&self) -> String {
        self.rust_info.version(self, channel::CFG_RELEASE_NUM)
    }

    /// Return the full commit hash
    fn rust_sha(&self) -> Option<&str> {
        self.rust_info.sha()
    }

    /// Returns the `a.b.c` version that the given package is at.
    fn release_num(&self, package: &str) -> String {
        let mut toml = String::new();
        let toml_file_name = self.src.join(&format!("src/tools/{}/Cargo.toml", package));
        t!(t!(File::open(toml_file_name)).read_to_string(&mut toml));
        for line in toml.lines() {
            let prefix = "version = \"";
            let suffix = "\"";
            if line.starts_with(prefix) && line.ends_with(suffix) {
                return line[prefix.len()..line.len() - suffix.len()].to_string()
            }
        }

        panic!("failed to find version in {}'s Cargo.toml", package)
    }

    /// Returns whether unstable features should be enabled for the compiler
    /// we're building.
    fn unstable_features(&self) -> bool {
        match &self.config.channel[..] {
            "stable" | "beta" => false,
            "nightly" | _ => true,
        }
    }

    /// Fold the output of the commands after this method into a group. The fold
    /// ends when the returned object is dropped. Folding can only be used in
    /// the Travis CI environment.
    pub fn fold_output<D, F>(&self, name: F) -> Option<OutputFolder>
        where D: Into<String>, F: FnOnce() -> D
    {
        if self.ci_env == CiEnv::Travis {
            Some(OutputFolder::new(name().into()))
        } else {
            None
        }
    }

    /// Updates the actual toolstate of a tool.
    ///
    /// The toolstates are saved to the file specified by the key
    /// `rust.save-toolstates` in `config.toml`. If unspecified, nothing will be
    /// done. The file is updated immediately after this function completes.
    pub fn save_toolstate(&self, tool: &str, state: ToolState) {
        use std::io::{Seek, SeekFrom};

        if let Some(ref path) = self.config.save_toolstates {
            let mut file = t!(fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(path));

            let mut current_toolstates: HashMap<Box<str>, ToolState> =
                serde_json::from_reader(&mut file).unwrap_or_default();
            current_toolstates.insert(tool.into(), state);
            t!(file.seek(SeekFrom::Start(0)));
            t!(file.set_len(0));
            t!(serde_json::to_writer(file, &current_toolstates));
        }
    }

    /// Get a list of crates from a root crate.
    ///
    /// Returns Vec<(crate, path to crate, is_root_crate)>
    fn crates(&self, root: &str) -> Vec<(Interned<String>, &Path)> {
        let interned = INTERNER.intern_string(root.to_owned());
        let mut ret = Vec::new();
        let mut list = vec![interned];
        let mut visited = HashSet::new();
        while let Some(krate) = list.pop() {
            let krate = &self.crates[&krate];
            // If we can't strip prefix, then out-of-tree path
            let path = krate.path.strip_prefix(&self.src).unwrap_or(&krate.path);
            ret.push((krate.name, path));
            for dep in &krate.deps {
                if visited.insert(dep) && dep != "build_helper" {
                    list.push(*dep);
                }
            }
        }
        ret
    }
}

impl<'a> Compiler {
    pub fn with_stage(mut self, stage: u32) -> Compiler {
        self.stage = stage;
        self
    }

    /// Returns whether this is a snapshot compiler for `build`'s configuration
    pub fn is_snapshot(&self, build: &Build) -> bool {
        self.stage == 0 && self.host == build.build
    }

    /// Returns if this compiler should be treated as a final stage one in the
    /// current build session.
    /// This takes into account whether we're performing a full bootstrap or
    /// not; don't directly compare the stage with `2`!
    pub fn is_final_stage(&self, build: &Build) -> bool {
        let final_stage = if build.config.full_bootstrap { 2 } else { 1 };
        self.stage >= final_stage
    }
}
