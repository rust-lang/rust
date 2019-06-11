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
//! When you execute `x.py build`, the steps executed are:
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
//! `build/$HOST/stage0-sysroot/lib/rustlib/$ARCH/lib`. FIXME: this step's
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

#![deny(rust_2018_idioms)]
#![deny(warnings)]
#![feature(core_intrinsics)]
#![feature(drain_filter)]

use std::cell::{RefCell, Cell};
use std::collections::{HashSet, HashMap};
use std::env;
use std::fs::{self, OpenOptions, File};
use std::io::{Seek, SeekFrom, Write, Read};
use std::path::{PathBuf, Path};
use std::process::{self, Command};
use std::slice;
use std::str;

#[cfg(unix)]
use std::os::unix::fs::symlink as symlink_file;
#[cfg(windows)]
use std::os::windows::fs::symlink_file;

use build_helper::{
    mtime, output, run_silent, run_suppressed, t, try_run_silent, try_run_suppressed,
};
use filetime::FileTime;

use crate::util::{exe, libdir, OutputFolder, CiEnv};

mod cc_detect;
mod channel;
mod check;
mod test;
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

#[cfg(all(unix, not(target_os = "haiku")))]
mod job {
    pub unsafe fn setup(build: &mut crate::Build) {
        if build.config.low_priority {
            libc::setpriority(libc::PRIO_PGRP as _, 0, 10);
        }
    }
}

#[cfg(any(target_os = "haiku", not(any(unix, windows))))]
mod job {
    pub unsafe fn setup(_build: &mut crate::Build) {
    }
}

pub use crate::config::Config;
use crate::flags::Subcommand;
use crate::cache::{Interned, INTERNER};
use crate::toolstate::ToolState;

const LLVM_TOOLS: &[&str] = &[
    "llvm-nm", // used to inspect binaries; it shows symbol names, their sizes and visibility
    "llvm-objcopy", // used to transform ELFs into binary format which flashing tools consume
    "llvm-objdump", // used to disassemble programs
    "llvm-profdata", // used to inspect and merge files generated by profiles
    "llvm-readobj", // used to get information from ELFs/objects that the other tools don't provide
    "llvm-size", // used to prints the size of the linker sections of a program
    "llvm-strip", // used to discard symbols from binary files to reduce their size
    "llvm-ar" // used for creating and modifying archive files
];

/// A structure representing a Rust compiler.
///
/// Each compiler has a `stage` that it is associated with and a `host` that
/// corresponds to the platform the compiler runs on. This structure is used as
/// a parameter to many methods below.
#[derive(Eq, PartialOrd, Ord, PartialEq, Clone, Copy, Hash, Debug)]
pub struct Compiler {
    stage: u32,
    host: Interned<String>,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum DocTests {
    // Default, run normal tests and doc tests.
    Yes,
    // Do not run any doc tests.
    No,
    // Only run doc tests.
    Only,
}

pub enum GitRepo {
    Rustc,
    Llvm,
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
    clippy_info: channel::GitInfo,
    miri_info: channel::GitInfo,
    rustfmt_info: channel::GitInfo,
    in_tree_llvm_info: channel::GitInfo,
    emscripten_llvm_info: channel::GitInfo,
    local_rebuild: bool,
    fail_fast: bool,
    doc_tests: DocTests,
    verbosity: usize,

    // Targets for which to build.
    build: Interned<String>,
    hosts: Vec<Interned<String>>,
    targets: Vec<Interned<String>>,

    // Stage 0 (downloaded) compiler and cargo or their local rust equivalents.
    initial_rustc: PathBuf,
    initial_cargo: PathBuf,

    // Runtime state filled in later on
    // C/C++ compilers and archiver for all targets
    cc: HashMap<Interned<String>, cc::Tool>,
    cxx: HashMap<Interned<String>, cc::Tool>,
    ar: HashMap<Interned<String>, PathBuf>,
    ranlib: HashMap<Interned<String>, PathBuf>,
    // Misc
    crates: HashMap<Interned<String>, Crate>,
    is_sudo: bool,
    ci_env: CiEnv,
    delayed_failures: RefCell<Vec<String>>,
    prerelease_version: Cell<Option<u32>>,
    tool_artifacts: RefCell<HashMap<
        Interned<String>,
        HashMap<String, (&'static str, PathBuf, Vec<String>)>
    >>,
}

#[derive(Debug)]
struct Crate {
    name: Interned<String>,
    deps: HashSet<Interned<String>>,
    id: String,
    path: PathBuf,
}

impl Crate {
    fn is_local(&self, build: &Build) -> bool {
        self.path.starts_with(&build.config.src) &&
        !self.path.to_string_lossy().ends_with("_shim")
    }

    fn local_path(&self, build: &Build) -> PathBuf {
        assert!(self.is_local(build));
        self.path.strip_prefix(&build.config.src).unwrap().into()
    }
}

/// The various "modes" of invoking Cargo.
///
/// These entries currently correspond to the various output directories of the
/// build system, with each mod generating output in a different directory.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    /// Build the standard library, placing output in the "stageN-std" directory.
    Std,

    /// Build libtest, placing output in the "stageN-test" directory.
    Test,

    /// Build librustc, and compiler libraries, placing output in the "stageN-rustc" directory.
    Rustc,

    /// Build codegen libraries, placing output in the "stageN-codegen" directory
    Codegen,

    /// Build some tools, placing output in the "stageN-tools" directory. The
    /// "other" here is for miscellaneous sets of tools that are built using the
    /// bootstrap compiler in its entirety (target libraries and all).
    /// Typically these tools compile with stable Rust.
    ToolBootstrap,

    /// Compile a tool which uses all libraries we compile (up to rustc).
    /// Doesn't use the stage0 compiler libraries like "other", and includes
    /// tools like rustdoc, cargo, rls, etc.
    ToolTest,
    ToolStd,
    ToolRustc,
}

impl Mode {
    pub fn is_tool(&self) -> bool {
        match self {
            Mode::ToolBootstrap | Mode::ToolRustc | Mode::ToolStd => true,
            _ => false
        }
    }
}

impl Build {
    /// Creates a new set of build configuration from the `flags` on the command
    /// line and the filesystem `config`.
    ///
    /// By default all build output will be placed in the current directory.
    pub fn new(config: Config) -> Build {
        let src = config.src.clone();
        let out = config.out.clone();

        let is_sudo = match env::var_os("SUDO_USER") {
            Some(sudo_user) => {
                match env::var_os("USER") {
                    Some(user) => user != sudo_user,
                    None => false,
                }
            }
            None => false,
        };

        let ignore_git = config.ignore_git;
        let rust_info = channel::GitInfo::new(ignore_git, &src);
        let cargo_info = channel::GitInfo::new(ignore_git, &src.join("src/tools/cargo"));
        let rls_info = channel::GitInfo::new(ignore_git, &src.join("src/tools/rls"));
        let clippy_info = channel::GitInfo::new(ignore_git, &src.join("src/tools/clippy"));
        let miri_info = channel::GitInfo::new(ignore_git, &src.join("src/tools/miri"));
        let rustfmt_info = channel::GitInfo::new(ignore_git, &src.join("src/tools/rustfmt"));

        // we always try to use git for LLVM builds
        let in_tree_llvm_info = channel::GitInfo::new(false, &src.join("src/llvm-project"));
        let emscripten_llvm_info = channel::GitInfo::new(false, &src.join("src/llvm-emscripten"));

        let mut build = Build {
            initial_rustc: config.initial_rustc.clone(),
            initial_cargo: config.initial_cargo.clone(),
            local_rebuild: config.local_rebuild,
            fail_fast: config.cmd.fail_fast(),
            doc_tests: config.cmd.doc_tests(),
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
            clippy_info,
            miri_info,
            rustfmt_info,
            in_tree_llvm_info,
            emscripten_llvm_info,
            cc: HashMap::new(),
            cxx: HashMap::new(),
            ar: HashMap::new(),
            ranlib: HashMap::new(),
            crates: HashMap::new(),
            is_sudo,
            ci_env: CiEnv::current(),
            delayed_failures: RefCell::new(Vec::new()),
            prerelease_version: Cell::new(None),
            tool_artifacts: Default::default(),
        };

        build.verbose("finding compilers");
        cc_detect::find(&mut build);
        build.verbose("running sanity check");
        sanity::check(&mut build);

        // If local-rust is the same major.minor as the current version, then force a
        // local-rebuild
        let local_version_verbose = output(
            Command::new(&build.initial_rustc).arg("--version").arg("--verbose"));
        let local_release = local_version_verbose
            .lines().filter(|x| x.starts_with("release:"))
            .next().unwrap().trim_start_matches("release:").trim();
        let my_version = channel::CFG_RELEASE_NUM;
        if local_release.split('.').take(2).eq(my_version.split('.').take(2)) {
            build.verbose(&format!("auto-detected local-rebuild {}", local_release));
            build.local_rebuild = true;
        }

        build.verbose("learning about cargo");
        metadata::build(&mut build);

        build
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

        {
            let builder = builder::Builder::new(&self);
            if let Some(path) = builder.paths.get(0) {
                if path == Path::new("nonexistent/path/to/trigger/cargo/metadata") {
                    return;
                }
            }
        }

        if !self.config.dry_run {
            {
                self.config.dry_run = true;
                let builder = builder::Builder::new(&self);
                builder.execute_cli();
            }
            self.config.dry_run = false;
            let builder = builder::Builder::new(&self);
            builder.execute_cli();
        } else {
            let builder = builder::Builder::new(&self);
            let _ = builder.execute_cli();
        }

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

    /// Gets the space-separated set of activated features for the standard
    /// library.
    fn std_features(&self) -> String {
        let mut features = "panic-unwind".to_string();

        if self.config.llvm_libunwind {
            features.push_str(" llvm-libunwind");
        }
        if self.config.backtrace {
            features.push_str(" backtrace");
        }
        if self.config.profiler {
            features.push_str(" profiler");
        }
        if self.config.wasm_syscall {
            features.push_str(" wasm_syscall");
        }
        features
    }

    /// Gets the space-separated set of activated features for the compiler.
    fn rustc_features(&self) -> String {
        let mut features = String::new();
        if self.config.jemalloc {
            features.push_str("jemalloc");
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

    /// Returns the root directory for all output generated in a particular
    /// stage when running with a particular host compiler.
    ///
    /// The mode indicates what the root directory is for.
    fn stage_out(&self, compiler: Compiler, mode: Mode) -> PathBuf {
        let suffix = match mode {
            Mode::Std => "-std",
            Mode::Test => "-test",
            Mode::Rustc => "-rustc",
            Mode::Codegen => "-codegen",
            Mode::ToolBootstrap => "-bootstrap-tools",
            Mode::ToolStd => "-tools",
            Mode::ToolTest => "-tools",
            Mode::ToolRustc => "-tools",
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

    fn emscripten_llvm_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("llvm-emscripten")
    }

    fn lld_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("lld")
    }

    /// Output directory for all documentation for a target
    fn doc_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("doc")
    }

    /// Output directory for all documentation for a target
    fn compiler_doc_out(&self, target: Interned<String>) -> PathBuf {
        self.out.join(&*target).join("compiler-doc")
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

    /// Returns `true` if no custom `llvm-config` is set for the specified target.
    ///
    /// If no custom `llvm-config` was specified then Rust's llvm will be used.
    fn is_rust_llvm(&self, target: Interned<String>) -> bool {
        match self.config.target_config.get(&target) {
            Some(ref c) => c.llvm_config.is_none(),
            None => true
        }
    }

    /// Returns the path to `FileCheck` binary for the specified target
    fn llvm_filecheck(&self, target: Interned<String>) -> PathBuf {
        let target_config = self.config.target_config.get(&target);
        if let Some(s) = target_config.and_then(|c| c.llvm_filecheck.as_ref()) {
            s.to_path_buf()
        } else if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            let llvm_bindir = output(Command::new(s).arg("--bindir"));
            let filecheck = Path::new(llvm_bindir.trim()).join(exe("FileCheck", &*target));
            if filecheck.exists() {
                filecheck
            } else {
                // On Fedora the system LLVM installs FileCheck in the
                // llvm subdirectory of the libdir.
                let llvm_libdir = output(Command::new(s).arg("--libdir"));
                let lib_filecheck = Path::new(llvm_libdir.trim())
                    .join("llvm").join(exe("FileCheck", &*target));
                if lib_filecheck.exists() {
                    lib_filecheck
                } else {
                    // Return the most normal file name, even though
                    // it doesn't exist, so that any error message
                    // refers to that.
                    filecheck
                }
            }
        } else {
            let base = self.llvm_out(self.config.build).join("build");
            let base = if !self.config.ninja && self.config.build.contains("msvc") {
                if self.config.llvm_optimize {
                    if self.config.llvm_release_debuginfo {
                        base.join("RelWithDebInfo")
                    } else {
                        base.join("Release")
                    }
                } else {
                    base.join("Debug")
                }
            } else {
                base
            };
            base.join("bin").join(exe("FileCheck", &*target))
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
        self.rustc_snapshot_sysroot().join(libdir(&self.config.build))
    }

    /// Returns the sysroot of the snapshot compiler.
    fn rustc_snapshot_sysroot(&self) -> &Path {
        self.initial_rustc.parent().unwrap().parent().unwrap()
    }

    /// Runs a command, printing out nice contextual information if it fails.
    fn run(&self, cmd: &mut Command) {
        if self.config.dry_run { return; }
        self.verbose(&format!("running: {:?}", cmd));
        run_silent(cmd)
    }

    /// Runs a command, printing out nice contextual information if it fails.
    fn run_quiet(&self, cmd: &mut Command) {
        if self.config.dry_run { return; }
        self.verbose(&format!("running: {:?}", cmd));
        run_suppressed(cmd)
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    fn try_run(&self, cmd: &mut Command) -> bool {
        if self.config.dry_run { return true; }
        self.verbose(&format!("running: {:?}", cmd));
        try_run_silent(cmd)
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    fn try_run_quiet(&self, cmd: &mut Command) -> bool {
        if self.config.dry_run { return true; }
        self.verbose(&format!("running: {:?}", cmd));
        try_run_suppressed(cmd)
    }

    pub fn is_verbose(&self) -> bool {
        self.verbosity > 0
    }

    /// Prints a message if this build is configured in verbose mode.
    fn verbose(&self, msg: &str) {
        if self.is_verbose() {
            println!("{}", msg);
        }
    }

    pub fn is_verbose_than(&self, level: usize) -> bool {
        self.verbosity > level
    }

    /// Prints a message if this build is configured in more verbose mode than `level`.
    fn verbose_than(&self, level: usize, msg: &str) {
        if self.is_verbose_than(level) {
            println!("{}", msg);
        }
    }

    fn info(&self, msg: &str) {
        if self.config.dry_run { return; }
        println!("{}", msg);
    }

    /// Returns the number of parallel jobs that have been configured for this
    /// build.
    fn jobs(&self) -> u32 {
        self.config.jobs.unwrap_or_else(|| num_cpus::get() as u32)
    }

    fn debuginfo_map(&self, which: GitRepo) -> Option<String> {
        if !self.config.rust_remap_debuginfo {
            return None
        }

        let path = match which {
            GitRepo::Rustc => {
                let sha = self.rust_sha().unwrap_or(channel::CFG_RELEASE_NUM);
                format!("/rustc/{}", sha)
            }
            GitRepo::Llvm => String::from("/rustc/llvm"),
        };
        Some(format!("{}={}", self.src.display(), path))
    }

    /// Returns the path to the C compiler for the target specified.
    fn cc(&self, target: Interned<String>) -> &Path {
        self.cc[&target].path()
    }

    /// Returns a list of flags to pass to the C compiler for the target
    /// specified.
    fn cflags(&self, target: Interned<String>, which: GitRepo) -> Vec<String> {
        // Filter out -O and /O (the optimization flags) that we picked up from
        // cc-rs because the build scripts will determine that for themselves.
        let mut base = self.cc[&target].args().iter()
                           .map(|s| s.to_string_lossy().into_owned())
                           .filter(|s| !s.starts_with("-O") && !s.starts_with("/O"))
                           .collect::<Vec<String>>();

        // If we're compiling on macOS then we add a few unconditional flags
        // indicating that we want libc++ (more filled out than libstdc++) and
        // we want to compile for 10.7. This way we can ensure that
        // LLVM/etc are all properly compiled.
        if target.contains("apple-darwin") {
            base.push("-stdlib=libc++".into());
        }

        // Work around an apparently bad MinGW / GCC optimization,
        // See: http://lists.llvm.org/pipermail/cfe-dev/2016-December/051980.html
        // See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=78936
        if &*target == "i686-pc-windows-gnu" {
            base.push("-fno-omit-frame-pointer".into());
        }

        if let Some(map) = self.debuginfo_map(which) {
        let cc = self.cc(target);
            if cc.ends_with("clang") || cc.ends_with("gcc") {
                base.push(format!("-fdebug-prefix-map={}", map));
            } else if cc.ends_with("clang-cl.exe") {
                base.push("-Xclang".into());
                base.push(format!("-fdebug-prefix-map={}", map));
            }
        }
        base
    }

    /// Returns the path to the `ar` archive utility for the target specified.
    fn ar(&self, target: Interned<String>) -> Option<&Path> {
        self.ar.get(&target).map(|p| &**p)
    }

    /// Returns the path to the `ranlib` utility for the target specified.
    fn ranlib(&self, target: Interned<String>) -> Option<&Path> {
        self.ranlib.get(&target).map(|p| &**p)
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

    /// Returns the path to the linker for the given target if it needs to be overridden.
    fn linker(&self, target: Interned<String>) -> Option<&Path> {
        if let Some(linker) = self.config.target_config.get(&target)
                                                       .and_then(|c| c.linker.as_ref()) {
            Some(linker)
        } else if target != self.config.build &&
                  !target.contains("msvc") &&
                  !target.contains("emscripten") &&
                  !target.contains("wasm32") &&
                  !target.contains("nvptx") &&
                  !target.contains("fuchsia") {
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

    /// Returns the sysroot for the wasi target, if defined
    fn wasi_root(&self, target: Interned<String>) -> Option<&Path> {
        self.config.target_config.get(&target)
            .and_then(|t| t.wasi_root.as_ref())
            .map(|p| &**p)
    }

    /// Returns `true` if this is a no-std `target`, if defined
    fn no_std(&self, target: Interned<String>) -> Option<bool> {
        self.config.target_config.get(&target)
            .map(|t| t.no_std)
    }

    /// Returns `true` if the target will be tested using the `remote-test-client`
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

    /// Given `num` in the form "a.b.c" return a "release string" which
    /// describes the release version number.
    ///
    /// For example on nightly this returns "a.b.c-nightly", on beta it returns
    /// "a.b.c-beta.1" and on stable it just returns "a.b.c".
    fn release(&self, num: &str) -> String {
        match &self.config.channel[..] {
            "stable" => num.to_string(),
            "beta" => if self.rust_info.is_git() {
                format!("{}-beta.{}", num, self.beta_prerelease_version())
            } else {
                format!("{}-beta", num)
            },
            "nightly" => format!("{}-nightly", num),
            _ => format!("{}-dev", num),
        }
    }

    fn beta_prerelease_version(&self) -> u32 {
        if let Some(s) = self.prerelease_version.get() {
            return s
        }

        let beta = output(
            Command::new("git")
                .arg("ls-remote")
                .arg("origin")
                .arg("beta")
                .current_dir(&self.src)
        );
        let beta = beta.trim().split_whitespace().next().unwrap();
        let master = output(
            Command::new("git")
                .arg("ls-remote")
                .arg("origin")
                .arg("master")
                .current_dir(&self.src)
        );
        let master = master.trim().split_whitespace().next().unwrap();

        // Figure out where the current beta branch started.
        let base = output(
            Command::new("git")
                .arg("merge-base")
                .arg(beta)
                .arg(master)
                .current_dir(&self.src),
        );
        let base = base.trim();

        // Next figure out how many merge commits happened since we branched off
        // beta. That's our beta number!
        let count = output(
            Command::new("git")
                .arg("rev-list")
                .arg("--count")
                .arg("--merges")
                .arg(format!("{}...HEAD", base))
                .current_dir(&self.src),
        );
        let n = count.trim().parse().unwrap();
        self.prerelease_version.set(Some(n));
        n
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

    /// Returns the value of `package_vers` above for clippy
    fn clippy_package_vers(&self) -> String {
        self.package_vers(&self.release_num("clippy"))
    }

    /// Returns the value of `package_vers` above for miri
    fn miri_package_vers(&self) -> String {
        self.package_vers(&self.release_num("miri"))
    }

    /// Returns the value of `package_vers` above for rustfmt
    fn rustfmt_package_vers(&self) -> String {
        self.package_vers(&self.release_num("rustfmt"))
    }

    fn llvm_tools_package_vers(&self) -> String {
        self.package_vers(channel::CFG_RELEASE_NUM)
    }

    fn llvm_tools_vers(&self) -> String {
        self.rust_version()
    }

    fn lldb_package_vers(&self) -> String {
        self.package_vers(channel::CFG_RELEASE_NUM)
    }

    fn lldb_vers(&self) -> String {
        self.rust_version()
    }

    fn llvm_link_tools_dynamically(&self, target: Interned<String>) -> bool {
        (target.contains("linux-gnu") || target.contains("apple-darwin"))
    }

    /// Returns the `version` string associated with this compiler for Rust
    /// itself.
    ///
    /// Note that this is a descriptive string which includes the commit date,
    /// sha, version, etc.
    fn rust_version(&self) -> String {
        self.rust_info.version(self, channel::CFG_RELEASE_NUM)
    }

    /// Returns the full commit hash.
    fn rust_sha(&self) -> Option<&str> {
        self.rust_info.sha()
    }

    /// Returns the `a.b.c` version that the given package is at.
    fn release_num(&self, package: &str) -> String {
        let toml_file_name = self.src.join(&format!("src/tools/{}/Cargo.toml", package));
        let toml = t!(fs::read_to_string(&toml_file_name));
        for line in toml.lines() {
            let prefix = "version = \"";
            let suffix = "\"";
            if line.starts_with(prefix) && line.ends_with(suffix) {
                return line[prefix.len()..line.len() - suffix.len()].to_string()
            }
        }

        panic!("failed to find version in {}'s Cargo.toml", package)
    }

    /// Returns `true` if unstable features should be enabled for the compiler
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
        if !self.config.dry_run && self.ci_env == CiEnv::Travis {
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

    fn in_tree_crates(&self, root: &str) -> Vec<&Crate> {
        let mut ret = Vec::new();
        let mut list = vec![INTERNER.intern_str(root)];
        let mut visited = HashSet::new();
        while let Some(krate) = list.pop() {
            let krate = &self.crates[&krate];
            if krate.is_local(self) {
                ret.push(krate);
            }
            for dep in &krate.deps {
                if visited.insert(dep) && dep != "build_helper" {
                    list.push(*dep);
                }
            }
        }
        ret
    }

    fn read_stamp_file(&self, stamp: &Path) -> Vec<(PathBuf, bool)> {
        if self.config.dry_run {
            return Vec::new();
        }

        let mut paths = Vec::new();
        let contents = t!(fs::read(stamp));
        // This is the method we use for extracting paths from the stamp file passed to us. See
        // run_cargo for more information (in compile.rs).
        for part in contents.split(|b| *b == 0) {
            if part.is_empty() {
                continue
            }
            let host = part[0] as char == 'h';
            let path = PathBuf::from(t!(str::from_utf8(&part[1..])));
            paths.push((path, host));
        }
        paths
    }

    /// Copies a file from `src` to `dst`
    pub fn copy(&self, src: &Path, dst: &Path) {
        if self.config.dry_run { return; }
        self.verbose_than(1, &format!("Copy {:?} to {:?}", src, dst));
        let _ = fs::remove_file(&dst);
        let metadata = t!(src.symlink_metadata());
        if metadata.file_type().is_symlink() {
            let link = t!(fs::read_link(src));
            t!(symlink_file(link, dst));
        } else if let Ok(()) = fs::hard_link(src, dst) {
            // Attempt to "easy copy" by creating a hard link
            // (symlinks don't work on windows), but if that fails
            // just fall back to a slow `copy` operation.
        } else {
            if let Err(e) = fs::copy(src, dst) {
                panic!("failed to copy `{}` to `{}`: {}", src.display(),
                       dst.display(), e)
            }
            t!(fs::set_permissions(dst, metadata.permissions()));
            let atime = FileTime::from_last_access_time(&metadata);
            let mtime = FileTime::from_last_modification_time(&metadata);
            t!(filetime::set_file_times(dst, atime, mtime));
        }
    }

    /// Search-and-replaces within a file. (Not maximally efficiently: allocates a
    /// new string for each replacement.)
    pub fn replace_in_file(&self, path: &Path, replacements: &[(&str, &str)]) {
        if self.config.dry_run { return; }
        let mut contents = String::new();
        let mut file = t!(OpenOptions::new().read(true).write(true).open(path));
        t!(file.read_to_string(&mut contents));
        for &(target, replacement) in replacements {
            contents = contents.replace(target, replacement);
        }
        t!(file.seek(SeekFrom::Start(0)));
        t!(file.set_len(0));
        t!(file.write_all(contents.as_bytes()));
    }

    /// Copies the `src` directory recursively to `dst`. Both are assumed to exist
    /// when this function is called.
    pub fn cp_r(&self, src: &Path, dst: &Path) {
        if self.config.dry_run { return; }
        for f in self.read_dir(src) {
            let path = f.path();
            let name = path.file_name().unwrap();
            let dst = dst.join(name);
            if t!(f.file_type()).is_dir() {
                t!(fs::create_dir_all(&dst));
                self.cp_r(&path, &dst);
            } else {
                let _ = fs::remove_file(&dst);
                self.copy(&path, &dst);
            }
        }
    }

    /// Copies the `src` directory recursively to `dst`. Both are assumed to exist
    /// when this function is called. Unwanted files or directories can be skipped
    /// by returning `false` from the filter function.
    pub fn cp_filtered(&self, src: &Path, dst: &Path, filter: &dyn Fn(&Path) -> bool) {
        // Immediately recurse with an empty relative path
        self.recurse_(src, dst, Path::new(""), filter)
    }

    // Inner function does the actual work
    fn recurse_(&self, src: &Path, dst: &Path, relative: &Path, filter: &dyn Fn(&Path) -> bool) {
        for f in self.read_dir(src) {
            let path = f.path();
            let name = path.file_name().unwrap();
            let dst = dst.join(name);
            let relative = relative.join(name);
            // Only copy file or directory if the filter function returns true
            if filter(&relative) {
                if t!(f.file_type()).is_dir() {
                    let _ = fs::remove_dir_all(&dst);
                    self.create_dir(&dst);
                    self.recurse_(&path, &dst, &relative, filter);
                } else {
                    let _ = fs::remove_file(&dst);
                    self.copy(&path, &dst);
                }
            }
        }
    }

    fn copy_to_folder(&self, src: &Path, dest_folder: &Path) {
        let file_name = src.file_name().unwrap();
        let dest = dest_folder.join(file_name);
        self.copy(src, &dest);
    }

    fn install(&self, src: &Path, dstdir: &Path, perms: u32) {
        if self.config.dry_run { return; }
        let dst = dstdir.join(src.file_name().unwrap());
        self.verbose_than(1, &format!("Install {:?} to {:?}", src, dst));
        t!(fs::create_dir_all(dstdir));
        drop(fs::remove_file(&dst));
        {
            if !src.exists() {
                panic!("Error: File \"{}\" not found!", src.display());
            }
            let metadata = t!(src.symlink_metadata());
            if let Err(e) = fs::copy(&src, &dst) {
                panic!("failed to copy `{}` to `{}`: {}", src.display(),
                       dst.display(), e)
            }
            t!(fs::set_permissions(&dst, metadata.permissions()));
            let atime = FileTime::from_last_access_time(&metadata);
            let mtime = FileTime::from_last_modification_time(&metadata);
            t!(filetime::set_file_times(&dst, atime, mtime));
        }
        chmod(&dst, perms);
    }

    fn create(&self, path: &Path, s: &str) {
        if self.config.dry_run { return; }
        t!(fs::write(path, s));
    }

    fn read(&self, path: &Path) -> String {
        if self.config.dry_run { return String::new(); }
        t!(fs::read_to_string(path))
    }

    fn create_dir(&self, dir: &Path) {
        if self.config.dry_run { return; }
        t!(fs::create_dir_all(dir))
    }

    fn remove_dir(&self, dir: &Path) {
        if self.config.dry_run { return; }
        t!(fs::remove_dir_all(dir))
    }

    fn read_dir(&self, dir: &Path) -> impl Iterator<Item=fs::DirEntry> {
        let iter = match fs::read_dir(dir) {
            Ok(v) => v,
            Err(_) if self.config.dry_run => return vec![].into_iter(),
            Err(err) => panic!("could not read dir {:?}: {:?}", dir, err),
        };
        iter.map(|e| t!(e)).collect::<Vec<_>>().into_iter()
    }

    fn remove(&self, f: &Path) {
        if self.config.dry_run { return; }
        fs::remove_file(f).unwrap_or_else(|_| panic!("failed to remove {:?}", f));
    }
}

#[cfg(unix)]
fn chmod(path: &Path, perms: u32) {
    use std::os::unix::fs::*;
    t!(fs::set_permissions(path, fs::Permissions::from_mode(perms)));
}
#[cfg(windows)]
fn chmod(_path: &Path, _perms: u32) {}


impl<'a> Compiler {
    pub fn with_stage(mut self, stage: u32) -> Compiler {
        self.stage = stage;
        self
    }

    /// Returns `true` if this is a snapshot compiler for `build`'s configuration
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
