// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::path::{PathBuf, Path};
use std::process::Command;

use build_helper::{run_silent, output};
use gcc;
use num_cpus;

use build::util::{exe, mtime, libdir, add_lib_path};

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

mod cc;
mod channel;
mod clean;
mod compile;
mod config;
mod doc;
mod flags;
mod native;
mod sanity;
mod step;
mod util;

#[cfg(windows)]
mod job;

#[cfg(not(windows))]
mod job {
    pub unsafe fn setup() {}
}

pub use build::config::Config;
pub use build::flags::Flags;

#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct Compiler<'a> {
    stage: u32,
    host: &'a str,
}

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
    bootstrap_key: String,

    // Runtime state filled in later on
    cc: HashMap<String, (gcc::Tool, PathBuf)>,
    cxx: HashMap<String, gcc::Tool>,
    compiler_rt_built: RefCell<HashMap<String, PathBuf>>,
}

impl Build {
    pub fn new(flags: Flags, config: Config) -> Build {
        let cwd = t!(env::current_dir());
        let src = flags.src.clone().unwrap_or(cwd.clone());
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
            bootstrap_key: String::new(),
            cc: HashMap::new(),
            cxx: HashMap::new(),
            compiler_rt_built: RefCell::new(HashMap::new()),
        }
    }

    pub fn build(&mut self) {
        use build::step::Source::*;

        unsafe {
            job::setup();
        }

        if self.flags.clean {
            return clean::clean(self);
        }

        cc::find(self);
        sanity::check(self);
        channel::collect(self);
        self.update_submodules();

        for target in step::all(self) {
            let doc_out = self.out.join(&target.target).join("doc");
            match target.src {
                Llvm { _dummy } => {
                    native::llvm(self, target.target);
                }
                CompilerRt { _dummy } => {
                    native::compiler_rt(self, target.target);
                }
                Libstd { stage, compiler } => {
                    compile::std(self, stage, target.target, &compiler);
                }
                Librustc { stage, compiler } => {
                    compile::rustc(self, stage, target.target, &compiler);
                }
                LibstdLink { stage, compiler, host } => {
                    compile::std_link(self, stage, target.target,
                                      &compiler, host);
                }
                LibrustcLink { stage, compiler, host } => {
                    compile::rustc_link(self, stage, target.target,
                                        &compiler, host);
                }
                Rustc { stage: 0 } => {
                    // nothing to do...
                }
                Rustc { stage } => {
                    compile::assemble_rustc(self, stage, target.target);
                }
                DocBook { stage } => {
                    doc::rustbook(self, stage, target.target, "book", &doc_out);
                }
                DocNomicon { stage } => {
                    doc::rustbook(self, stage, target.target, "nomicon",
                                  &doc_out);
                }
                DocStyle { stage } => {
                    doc::rustbook(self, stage, target.target, "style",
                                  &doc_out);
                }
                DocStandalone { stage } => {
                    doc::standalone(self, stage, target.target, &doc_out);
                }
                Doc { .. } => {} // pseudo-step
            }
        }
    }

    fn update_submodules(&self) {
        if !self.config.submodules {
            return
        }
        if fs::metadata(self.src.join(".git")).is_err() {
            return
        }
        let git_submodule = || {
            let mut cmd = Command::new("git");
            cmd.current_dir(&self.src).arg("submodule");
            return cmd
        };
        let out = output(git_submodule().arg("status"));
        if !out.lines().any(|l| l.starts_with("+") || l.starts_with("-")) {
            return
        }

        self.run(git_submodule().arg("sync"));
        self.run(git_submodule().arg("init"));
        self.run(git_submodule().arg("update"));
        self.run(git_submodule().arg("update").arg("--recursive"));
        self.run(git_submodule().arg("status").arg("--recursive"));
        self.run(git_submodule().arg("foreach").arg("--recursive")
                                .arg("git").arg("clean").arg("-fdx"));
        self.run(git_submodule().arg("foreach").arg("--recursive")
                                .arg("git").arg("checkout").arg("."));
    }

    /// Clear out `dir` if our build has been flagged as dirty, and also set
    /// ourselves as dirty if `file` changes when `f` is executed.
    fn clear_if_dirty(&self, dir: &Path, input: &Path) {
        let stamp = dir.join(".stamp");
        if mtime(&stamp) < mtime(input) {
            self.verbose(&format!("Dirty - {}", dir.display()));
            let _ = fs::remove_dir_all(dir);
        }
        t!(fs::create_dir_all(dir));
        t!(File::create(stamp));
    }

    /// Prepares an invocation of `cargo` to be run.
    ///
    /// This will create a `Command` that represents a pending execution of
    /// Cargo for the specified stage, whether or not the standard library is
    /// being built, and using the specified compiler targeting `target`.
    // FIXME: aren't stage/compiler duplicated?
    fn cargo(&self, stage: u32, compiler: &Compiler, is_std: bool,
             target: &str, cmd: &str) -> Command {
        let mut cargo = Command::new(&self.cargo);
        let host = compiler.host;
        let out_dir = self.stage_out(stage, host, is_std);
        cargo.env("CARGO_TARGET_DIR", out_dir)
             .arg(cmd)
             .arg("--target").arg(target)
             .arg("-j").arg(self.jobs().to_string());

        // Customize the compiler we're running. Specify the compiler to cargo
        // as our shim and then pass it some various options used to configure
        // how the actual compiler itself is called.
        cargo.env("RUSTC", self.out.join("bootstrap/debug/rustc"))
             .env("RUSTC_REAL", self.compiler_path(compiler))
             .env("RUSTC_STAGE", self.stage_arg(stage, compiler).to_string())
             .env("RUSTC_DEBUGINFO", self.config.rust_debuginfo.to_string())
             .env("RUSTC_CODEGEN_UNITS",
                  self.config.rust_codegen_units.to_string())
             .env("RUSTC_DEBUG_ASSERTIONS",
                  self.config.rust_debug_assertions.to_string())
             .env("RUSTC_SNAPSHOT", &self.rustc)
             .env("RUSTC_SYSROOT", self.sysroot(stage, host))
             .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_snapshot_libdir())
             .env("RUSTC_FLAGS", self.rustc_flags(target).join(" "))
             .env("RUSTC_RPATH", self.config.rust_rpath.to_string())
             .env("RUSTDOC", self.tool(compiler, "rustdoc"));

        // Specify some variuos options for build scripts used throughout the
        // build.
        //
        // FIXME: the guard against msvc shouldn't need to be here
        if !target.contains("msvc") {
            cargo.env(format!("CC_{}", target), self.cc(target))
                 .env(format!("AR_{}", target), self.ar(target))
                 .env(format!("CFLAGS_{}", target), self.cflags(target));
        }

        // Environment variables *required* needed throughout the build
        //
        // FIXME: should update code to not require this env vars
        cargo.env("CFG_COMPILER_HOST_TRIPLE", target);

        if self.config.verbose || self.flags.verbose {
            cargo.arg("-v");
        }
        if self.config.rust_optimize {
            cargo.arg("--release");
        }
        self.add_rustc_lib_path(compiler, &mut cargo);
        return cargo
    }

    /// Get a path to the compiler specified.
    fn compiler_path(&self, compiler: &Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.rustc.clone()
        } else {
            self.sysroot(compiler.stage, compiler.host).join("bin")
                .join(exe("rustc", compiler.host))
        }
    }

    /// Get the specified tool next to the specified compiler
    fn tool(&self, compiler: &Compiler, tool: &str) -> PathBuf {
        if compiler.is_snapshot(self) {
            assert!(tool == "rustdoc", "no tools other than rustdoc in stage0");
            let mut rustdoc = self.rustc.clone();
            rustdoc.pop();
            rustdoc.push(exe("rustdoc", &self.config.build));
            return rustdoc
        }
        let (stage, host) = (compiler.stage, compiler.host);
        self.cargo_out(stage - 1, host, false, host).join(exe(tool, host))
    }

    /// Get a `Command` which is ready to run `tool` in `stage` built for
    /// `host`.
    #[allow(dead_code)] // this will be used soon
    fn tool_cmd(&self, compiler: &Compiler, tool: &str) -> Command {
        let mut cmd = Command::new(self.tool(&compiler, tool));
        let host = compiler.host;
        let stage = compiler.stage;
        let paths = vec![
            self.cargo_out(stage - 1, host, true, host).join("deps"),
            self.cargo_out(stage - 1, host, false, host).join("deps"),
        ];
        add_lib_path(paths, &mut cmd);
        return cmd
    }

    fn stage_arg(&self, stage: u32, compiler: &Compiler) -> u32 {
        if stage == 0 && compiler.host != self.config.build {1} else {stage}
    }

    /// Get the space-separated set of activated features for the standard
    /// library.
    fn std_features(&self) -> String {
        let mut features = String::new();
        if self.config.debug_jemalloc {
            features.push_str(" debug-jemalloc");
        }
        if self.config.use_jemalloc {
            features.push_str(" jemalloc");
        }
        return features
    }

    /// Get the space-separated set of activated features for the compiler.
    fn rustc_features(&self, stage: u32) -> String {
        let mut features = String::new();
        if self.config.use_jemalloc {
            features.push_str(" jemalloc");
        }
        if stage > 0 {
            features.push_str(" rustdoc");
            features.push_str(" rustbook");
        }
        return features
    }

    /// Component directory that Cargo will produce output into (e.g.
    /// release/debug)
    fn cargo_dir(&self) -> &'static str {
        if self.config.rust_optimize {"release"} else {"debug"}
    }

    fn sysroot(&self, stage: u32, host: &str) -> PathBuf {
        if stage == 0 {
            self.stage_out(stage, host, false)
        } else {
            self.out.join(host).join(format!("stage{}", stage))
        }
    }

    fn sysroot_libdir(&self, stage: u32, host: &str, target: &str) -> PathBuf {
        self.sysroot(stage, host).join("lib").join("rustlib")
            .join(target).join("lib")
    }

    /// Returns the root directory for all output generated in a particular
    /// stage when running with a particular host compiler.
    ///
    /// The `is_std` flag indicates whether the root directory is for the
    /// bootstrap of the standard library or for the compiler.
    fn stage_out(&self, stage: u32, host: &str, is_std: bool) -> PathBuf {
        self.out.join(host)
            .join(format!("stage{}{}", stage, if is_std {"-std"} else {"-rustc"}))
    }

    /// Returns the root output directory for all Cargo output in a given stage,
    /// running a particular comipler, wehther or not we're building the
    /// standard library, and targeting the specified architecture.
    fn cargo_out(&self, stage: u32, host: &str, is_std: bool,
                 target: &str) -> PathBuf {
        self.stage_out(stage, host, is_std).join(target).join(self.cargo_dir())
    }

    /// Root output directory for LLVM compiled for `target`
    fn llvm_out(&self, target: &str) -> PathBuf {
        self.out.join(target).join("llvm")
    }

    /// Root output directory for compiler-rt compiled for `target`
    fn compiler_rt_out(&self, target: &str) -> PathBuf {
        self.out.join(target).join("compiler-rt")
    }

    fn add_rustc_lib_path(&self, compiler: &Compiler, cmd: &mut Command) {
        // Windows doesn't need dylib path munging because the dlls for the
        // compiler live next to the compiler and the system will find them
        // automatically.
        if cfg!(windows) { return }

        add_lib_path(vec![self.rustc_libdir(compiler)], cmd);
    }

    fn rustc_libdir(&self, compiler: &Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.rustc_snapshot_libdir()
        } else {
            self.sysroot(compiler.stage, compiler.host)
                .join(libdir(compiler.host))
        }
    }

    fn rustc_snapshot_libdir(&self) -> PathBuf {
        self.rustc.parent().unwrap().parent().unwrap()
            .join(libdir(&self.config.build))
    }

    fn run(&self, cmd: &mut Command) {
        self.verbose(&format!("running: {:?}", cmd));
        run_silent(cmd)
    }

    fn verbose(&self, msg: &str) {
        if self.flags.verbose || self.config.verbose {
            println!("{}", msg);
        }
    }

    fn jobs(&self) -> u32 {
        self.flags.jobs.unwrap_or(num_cpus::get() as u32)
    }

    fn cc(&self, target: &str) -> &Path {
        self.cc[target].0.path()
    }

    fn cflags(&self, target: &str) -> String {
        self.cc[target].0.args().iter()
            .map(|s| s.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn ar(&self, target: &str) -> &Path {
        &self.cc[target].1
    }

    fn cxx(&self, target: &str) -> &Path {
        self.cxx[target].path()
    }

    fn rustc_flags(&self, target: &str) -> Vec<String> {
        let mut base = Vec::new();
        if target != self.config.build && !target.contains("msvc") {
            base.push(format!("-Clinker={}", self.cc(target).display()));
        }
        return base
    }
}

impl<'a> Compiler<'a> {
    fn new(stage: u32, host: &'a str) -> Compiler<'a> {
        Compiler { stage: stage, host: host }
    }

    fn is_snapshot(&self, build: &Build) -> bool {
        self.stage == 0 && self.host == build.config.build
    }
}
