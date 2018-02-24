// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::env;
use std::fmt::Debug;
use std::fs::{self, File};
use std::io::{BufRead, Read, Write, BufReader};
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use filetime::FileTime;
use serde_json;

use compile;
use install;
use dist;
use util::{add_lib_path, is_dylib, exe, libdir, CiEnv};
use build_helper::mtime;
use {Build, Mode};
use cache::{Cache, Intern, Interned};
use check;
use test;
use flags::Subcommand;
use doc;
use tool;
use native;

pub use Compiler;

pub struct Builder<'a> {
    pub build: &'a Build,
    pub top_stage: u32,
    pub kind: Kind,
    cache: Cache,
    stack: RefCell<Vec<Box<Any>>>,
}

impl<'a> Deref for Builder<'a> {
    type Target = Build;

    fn deref(&self) -> &Self::Target {
        self.build
    }
}

pub trait Step: 'static + Clone + Debug + PartialEq + Eq + Hash {
    /// `PathBuf` when directories are created or to return a `Compiler` once
    /// it's been assembled.
    type Output: Clone;

    const DEFAULT: bool = false;

    /// Run this rule for all hosts without cross compiling.
    const ONLY_HOSTS: bool = false;

    /// Primary function to execute this rule. Can call `builder.ensure(...)`
    /// with other steps to run those.
    fn run(self, builder: &Builder) -> Self::Output;

    /// When bootstrap is passed a set of paths, this controls whether this rule
    /// will execute. However, it does not get called in a "default" context
    /// when we are not passed any paths; in that case, make_run is called
    /// directly.
    fn should_run(run: ShouldRun) -> ShouldRun;

    /// Build up a "root" rule, either as a default rule or from a path passed
    /// to us.
    ///
    /// When path is `None`, we are executing in a context where no paths were
    /// passed. When `./x.py build` is run, for example, this rule could get
    /// called if it is in the correct list below with a path of `None`.
    fn make_run(_run: RunConfig) {
        // It is reasonable to not have an implementation of make_run for rules
        // who do not want to get called from the root context. This means that
        // they are likely dependencies (e.g., sysroot creation) or similar, and
        // as such calling them from ./x.py isn't logical.
        unimplemented!()
    }
}

pub struct RunConfig<'a> {
    pub builder: &'a Builder<'a>,
    pub host: Interned<String>,
    pub target: Interned<String>,
    pub path: PathBuf,
}

struct StepDescription {
    default: bool,
    only_hosts: bool,
    should_run: fn(ShouldRun) -> ShouldRun,
    make_run: fn(RunConfig),
    name: &'static str,
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
struct PathSet {
    set: BTreeSet<PathBuf>,
}

impl PathSet {
    fn empty() -> PathSet {
        PathSet {
            set: BTreeSet::new(),
        }
    }

    fn one<P: Into<PathBuf>>(path: P) -> PathSet {
        let mut set = BTreeSet::new();
        set.insert(path.into());
        PathSet { set }
    }

    fn has(&self, needle: &Path) -> bool {
        self.set.iter().any(|p| p.ends_with(needle))
    }

    fn path(&self, builder: &Builder) -> PathBuf {
        self.set
            .iter()
            .next()
            .unwrap_or(&builder.config.src)
            .to_path_buf()
    }
}

impl StepDescription {
    fn from<S: Step>() -> StepDescription {
        StepDescription {
            default: S::DEFAULT,
            only_hosts: S::ONLY_HOSTS,
            should_run: S::should_run,
            make_run: S::make_run,
            name: unsafe { ::std::intrinsics::type_name::<S>() },
        }
    }

    fn maybe_run(&self, builder: &Builder, pathset: &PathSet) {
        if builder.config.exclude.iter().any(|e| pathset.has(e)) {
            eprintln!("Skipping {:?} because it is excluded", pathset);
            return;
        } else if !builder.config.exclude.is_empty() {
            eprintln!(
                "{:?} not skipped for {:?} -- not in {:?}",
                pathset, self.name, builder.config.exclude
            );
        }
        let hosts = &builder.config.general.host;

        if self.only_hosts && !builder.config.run_host_only {
            return;
        }

        // Determine the targets participating in this rule.
        let targets = if self.only_hosts {
            &builder.config.general.host
        } else {
            &builder.config.general.target
        };

        for host in hosts {
            for target in targets {
                let run = RunConfig {
                    builder,
                    path: pathset.path(builder),
                    host: *host,
                    target: *target,
                };
                (self.make_run)(run);
            }
        }
    }
}

#[derive(Clone)]
pub struct ShouldRun<'a> {
    pub builder: &'a Builder<'a>,
    // use a BTreeSet to maintain sort order
    paths: BTreeSet<PathSet>,

    // If this is a default rule, this is an additional constraint placed on
    // it's run. Generally something like compiler docs being enabled.
    is_really_default: bool,
}

impl<'a> ShouldRun<'a> {
    fn new(builder: &'a Builder) -> ShouldRun<'a> {
        ShouldRun {
            builder,
            paths: BTreeSet::new(),
            is_really_default: true, // by default no additional conditions
        }
    }

    pub fn default_condition(mut self, cond: bool) -> Self {
        self.is_really_default = cond;
        self
    }

    // Unlike `krate` this will create just one pathset. As such, it probably shouldn't actually
    // ever be used, but as we transition to having all rules properly handle passing krate(...) by
    // actually doing something different for every crate passed.
    pub fn all_krates(mut self, name: &str) -> Self {
        let mut set = BTreeSet::new();
        for krate in self.builder.in_tree_crates(name) {
            set.insert(PathBuf::from(&krate.path));
        }
        self.paths.insert(PathSet { set });
        self
    }

    pub fn krate(mut self, name: &str) -> Self {
        for krate in self.builder.in_tree_crates(name) {
            self.paths.insert(PathSet::one(&krate.path));
        }
        self
    }

    // single, non-aliased path
    pub fn path(self, path: &str) -> Self {
        self.paths(&[path])
    }

    // multiple aliases for the same job
    pub fn paths(mut self, paths: &[&str]) -> Self {
        self.paths.insert(PathSet {
            set: paths.iter().map(PathBuf::from).collect(),
        });
        self
    }

    // allows being more explicit about why should_run in Step returns the value passed to it
    pub fn never(mut self) -> ShouldRun<'a> {
        self.paths.insert(PathSet::empty());
        self
    }

    fn pathset_for_path(&self, path: &Path) -> Option<&PathSet> {
        self.paths.iter().find(|pathset| pathset.has(path))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Kind {
    Build,
    Check,
    Test,
    Bench,
    Dist,
    Doc,
    Install,
}

impl<'a> Builder<'a> {
    pub fn run(build: &Build) {
        let kind = match build.config.cmd {
            Subcommand::Build => Kind::Build,
            Subcommand::Check => Kind::Check,
            Subcommand::Doc { .. } => Kind::Doc,
            Subcommand::Test { .. } => Kind::Test,
            Subcommand::Bench { .. } => Kind::Bench,
            Subcommand::Dist => Kind::Dist,
            Subcommand::Install { .. } => Kind::Install,
            Subcommand::Clean { .. } => panic!(),
        };

        if let Some(path) = build.config.paths.get(0) {
            if path == Path::new("nonexistent/path/to/trigger/cargo/metadata") {
                return;
            }
        }

        let builder = Builder {
            build,
            top_stage: build.config.stage,
            kind,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
        };

        if kind == Kind::Dist {
            assert!(
                !builder.config.rust.test_miri,
                "Do not distribute with miri enabled.\n\
                The distributed libraries would include all MIR (increasing binary size).
                The distributed MIR would include validation statements."
            );
        }
        builder.run_step_descriptions(&Builder::get_step_descriptions(kind), &builder.config.paths);
    }

    fn run_step_descriptions(&self, v: &[StepDescription], paths: &[PathBuf]) {
        let should_runs = v.iter()
            .map(|desc| (desc.should_run)(ShouldRun::new(self)))
            .collect::<Vec<_>>();

        // sanity checks on rules
        for (desc, should_run) in v.iter().zip(&should_runs) {
            assert!(
                !should_run.paths.is_empty(),
                "{:?} should have at least one pathset",
                desc.name
            );
        }

        if paths.is_empty() {
            for (desc, should_run) in v.iter().zip(should_runs) {
                if desc.default && should_run.is_really_default {
                    for pathset in &should_run.paths {
                        desc.maybe_run(&self, pathset);
                    }
                }
            }
        } else {
            for path in paths {
                let mut attempted_run = false;
                if path == Path::new("nonexistent/path/to/trigger/cargo/metadata") {
                    continue;
                }
                for (desc, should_run) in v.iter().zip(&should_runs) {
                    if let Some(pathset) = should_run.pathset_for_path(path) {
                        attempted_run = true;
                        desc.maybe_run(&self, pathset);
                    }
                }

                if !attempted_run {
                    panic!("Error: no rules matched {}.", path.display());
                }
            }
        }
    }

    fn get_step_descriptions(kind: Kind) -> Vec<StepDescription> {
        macro_rules! describe {
            ($($rule:ty),+ $(,)*) => {{
                vec![$(StepDescription::from::<$rule>()),+]
            }};
        }
        match kind {
            Kind::Build => describe!(
                compile::Std,
                compile::Test,
                compile::Rustc,
                compile::StartupObjects,
                tool::BuildManifest,
                tool::Rustbook,
                tool::ErrorIndex,
                tool::UnstableBookGen,
                tool::Tidy,
                tool::Linkchecker,
                tool::CargoTest,
                tool::Compiletest,
                tool::RemoteTestServer,
                tool::RemoteTestClient,
                tool::RustInstaller,
                tool::Cargo,
                tool::Rls,
                tool::Rustdoc,
                tool::Clippy,
                native::Llvm,
                tool::Rustfmt,
                tool::Miri
            ),
            Kind::Check => describe!(check::Std, check::Test, check::Rustc),
            Kind::Test => describe!(
                test::Tidy,
                test::Bootstrap,
                test::Ui,
                test::RunPass,
                test::CompileFail,
                test::ParseFail,
                test::RunFail,
                test::RunPassValgrind,
                test::MirOpt,
                test::Codegen,
                test::CodegenUnits,
                test::Incremental,
                test::Debuginfo,
                test::UiFullDeps,
                test::RunPassFullDeps,
                test::RunFailFullDeps,
                test::CompileFailFullDeps,
                test::IncrementalFullDeps,
                test::Rustdoc,
                test::Pretty,
                test::RunPassPretty,
                test::RunFailPretty,
                test::RunPassValgrindPretty,
                test::RunPassFullDepsPretty,
                test::RunFailFullDepsPretty,
                test::RunMake,
                test::Crate,
                test::CrateLibrustc,
                test::Rustdoc,
                test::Linkcheck,
                test::Cargotest,
                test::Cargo,
                test::Rls,
                test::Docs,
                test::ErrorIndex,
                test::Distcheck,
                test::Rustfmt,
                test::Miri,
                test::Clippy,
                test::RustdocJS,
                test::RustdocTheme
            ),
            Kind::Bench => describe!(test::Crate, test::CrateLibrustc),
            Kind::Doc => describe!(
                doc::UnstableBook,
                doc::UnstableBookGen,
                doc::TheBook,
                doc::Standalone,
                doc::Std,
                doc::Test,
                doc::Rustc,
                doc::ErrorIndex,
                doc::Nomicon,
                doc::Reference,
                doc::Rustdoc,
                doc::RustByExample,
                doc::CargoBook
            ),
            Kind::Dist => describe!(
                dist::Docs,
                dist::Mingw,
                dist::Rustc,
                dist::DebuggerScripts,
                dist::Std,
                dist::Analysis,
                dist::Src,
                dist::PlainSourceTarball,
                dist::Cargo,
                dist::Rls,
                dist::Rustfmt,
                dist::Extended,
                dist::HashSign
            ),
            Kind::Install => describe!(
                install::Docs,
                install::Std,
                install::Cargo,
                install::Rls,
                install::Rustfmt,
                install::Analysis,
                install::Src,
                install::Rustc
            ),
        }
    }

    pub fn get_help(build: &Build, subcommand: &str) -> Option<String> {
        let kind = match subcommand {
            "build" => Kind::Build,
            "doc" => Kind::Doc,
            "test" => Kind::Test,
            "bench" => Kind::Bench,
            "dist" => Kind::Dist,
            "install" => Kind::Install,
            _ => return None,
        };

        let builder = Builder {
            build,
            top_stage: build.config.stage,
            kind,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
        };

        let builder = &builder;
        let mut should_run = ShouldRun::new(builder);
        for desc in Builder::get_step_descriptions(builder.kind) {
            should_run = (desc.should_run)(should_run);
        }
        let mut help = String::from("Available paths:\n");
        for pathset in should_run.paths {
            for path in pathset.set {
                help.push_str(format!("    ./x.py {} {}\n", subcommand, path.display()).as_str());
            }
        }
        Some(help)
    }

    pub fn default_doc(&self, paths: Option<&[PathBuf]>) {
        let paths = paths.unwrap_or(&[]);
        self.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), paths);
    }

    /// Obtain a compiler at a given stage and for a given host. Explicitly does
    /// not take `Compiler` since all `Compiler` instances are meant to be
    /// obtained through this function, since it ensures that they are valid
    /// (i.e., built and assembled).
    pub fn compiler(&self, stage: u32, host: Interned<String>) -> Compiler {
        self.ensure(compile::Assemble {
            target_compiler: Compiler { stage, host },
        })
    }

    pub fn sysroot(&self, compiler: Compiler) -> Interned<PathBuf> {
        self.ensure(compile::Sysroot { compiler })
    }

    /// Returns the libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
    pub fn sysroot_libdir(
        &self,
        compiler: Compiler,
        target: Interned<String>,
    ) -> Interned<PathBuf> {
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        struct Libdir {
            compiler: Compiler,
            target: Interned<String>,
        }
        impl Step for Libdir {
            type Output = Interned<PathBuf>;

            fn should_run(run: ShouldRun) -> ShouldRun {
                run.never()
            }

            fn run(self, builder: &Builder) -> Interned<PathBuf> {
                let compiler = self.compiler;
                let config = &builder.build.config;
                let lib = if compiler.stage >= 1 && config.libdir_relative().is_some() {
                    config.libdir_relative().unwrap()
                } else {
                    Path::new("lib")
                };
                let sysroot = builder
                    .sysroot(self.compiler)
                    .join(lib)
                    .join("rustlib")
                    .join(self.target)
                    .join("lib");
                let _ = fs::remove_dir_all(&sysroot);
                t!(fs::create_dir_all(&sysroot));
                sysroot.intern()
            }
        }
        self.ensure(Libdir { compiler, target })
    }

    pub fn sysroot_codegen_backends(&self, compiler: Compiler) -> PathBuf {
        self.sysroot_libdir(compiler, compiler.host)
            .with_file_name("codegen-backends")
    }

    /// Returns the compiler's libdir where it stores the dynamic libraries that
    /// it itself links against.
    ///
    /// For example this returns `<sysroot>/lib` on Unix and `<sysroot>/bin` on
    /// Windows.
    pub fn rustc_libdir(&self, compiler: Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.rustc_snapshot_libdir()
        } else {
            self.sysroot(compiler).join(libdir(&compiler.host))
        }
    }

    /// Adds the compiler's directory of dynamic libraries to `cmd`'s dynamic
    /// library lookup path.
    pub fn add_rustc_lib_path(&self, compiler: Compiler, cmd: &mut Command) {
        // Windows doesn't need dylib path munging because the dlls for the
        // compiler live next to the compiler and the system will find them
        // automatically.
        if cfg!(windows) {
            return;
        }

        add_lib_path(vec![self.rustc_libdir(compiler)], cmd);
    }

    /// Get a path to the compiler specified.
    pub fn rustc(&self, compiler: Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.config.general.initial_rustc.clone()
        } else {
            self.sysroot(compiler)
                .join("bin")
                .join(exe("rustc", &compiler.host))
        }
    }

    pub fn rustdoc(&self, host: Interned<String>) -> PathBuf {
        self.ensure(tool::Rustdoc { host })
    }

    pub fn rustdoc_cmd(&self, host: Interned<String>) -> Command {
        let mut cmd = Command::new(&self.config.general.out.join("bootstrap/debug/rustdoc"));
        let compiler = self.compiler(self.top_stage, host);
        cmd.env("RUSTC_STAGE", compiler.stage.to_string())
            .env("RUSTC_SYSROOT", self.sysroot(compiler))
            .env(
                "RUSTDOC_LIBDIR",
                self.sysroot_libdir(compiler, self.config.general.build),
            )
            .env("CFG_RELEASE_CHANNEL", &self.config.rust.channel)
            .env("RUSTDOC_REAL", self.rustdoc(host))
            .env("RUSTDOC_CRATE_VERSION", self.rust_version())
            .env("RUSTC_BOOTSTRAP", "1");
        if let Some(linker) = self.linker(host) {
            cmd.env("RUSTC_TARGET_LINKER", linker);
        }
        cmd
    }

    /// Cargo's output path for the standard library in a given stage, compiled
    /// by a particular compiler for the specified target.
    pub fn libstd_stamp(&self, compiler: Compiler, target: Interned<String>) -> PathBuf {
        self
            .cargo_out(compiler, Mode::Libstd, target)
            .join(".libstd.stamp")
    }

    /// Cargo's output path for libtest in a given stage, compiled by a particular
    /// compiler for the specified target.
    pub fn libtest_stamp(&self, compiler: Compiler, target: Interned<String>) -> PathBuf {
        self
            .cargo_out(compiler, Mode::Libtest, target)
            .join(".libtest.stamp")
    }

    /// Cargo's output path for librustc in a given stage, compiled by a particular
    /// compiler for the specified target.
    pub fn librustc_stamp(&self, compiler: Compiler, target: Interned<String>) -> PathBuf {
        self
            .cargo_out(compiler, Mode::Librustc, target)
            .join(".librustc.stamp")
    }

    pub fn codegen_backend_stamp(
        &self,
        compiler: Compiler,
        target: Interned<String>,
        backend: &str,
    ) -> PathBuf {
        self
            .cargo_out(compiler, Mode::Librustc, target)
            .join(format!(".librustc_trans-{}.stamp", backend))
    }

    /// Configure cargo to compile the standard library, adding appropriate env vars
    /// and such.
    fn std_cargo(
        &self,
        compiler: Compiler,
        target: Interned<String>,
        cargo: &mut Command,
    ) {
        let mut features = self.std_features();

        if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
            cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
        }

        // When doing a local rebuild we tell cargo that we're stage1 rather than
        // stage0. This works fine if the local rust and being-built rust have the
        // same view of what the default allocator is, but fails otherwise. Since
        // we don't have a way to express an allocator preference yet, work
        // around the issue in the case of a local rebuild with jemalloc disabled.
        if compiler.stage == 0 && self.config.general.local_rebuild
            && !self.config.rust.use_jemalloc
        {
            features.push_str(" force_alloc_system");
        }

        if compiler.stage != 0 && self.config.general.sanitizers {
            // This variable is used by the sanitizer runtime crates, e.g.
            // rustc_lsan, to build the sanitizer runtime from C code
            // When this variable is missing, those crates won't compile the C code,
            // so we don't set this variable during stage0 where llvm-config is
            // missing
            // We also only build the runtimes when --enable-sanitizers (or its
            // config.toml equivalent) is used
            cargo.env("LLVM_CONFIG", self.llvm_config(target));
        }

        cargo
            .arg("--features")
            .arg(features)
            .arg("--manifest-path")
            .arg(self.config.src.join("src/libstd/Cargo.toml"));

        if let Some(target) = self.config.target_config.get(&target) {
            if let Some(ref jemalloc) = target.jemalloc {
                cargo.env("JEMALLOC_OVERRIDE", jemalloc);
            }
        }
        if target.contains("musl") {
            if let Some(p) = self.musl_root(target) {
                cargo.env("MUSL_ROOT", p);
            }
        }
    }

    /// Same as `std_cargo`, but for libtest
    fn test_cargo(&self, cargo: &mut Command) {
        if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
            cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
        }
        cargo
            .arg("--manifest-path")
            .arg(self.config.src.join("src/libtest/Cargo.toml"));
    }

    // A little different from {std,test}_cargo since we don't pass
    // --manifest-path or --features, since those vary between the
    // codegen backend building and normal rustc library building.
    fn rustc_cargo(&self, cargo: &mut Command) {
        // Set some configuration variables picked up by build scripts and
        // the compiler alike
        cargo
            .env("CFG_RELEASE", self.rust_release())
            .env("CFG_RELEASE_CHANNEL", &self.config.rust.channel)
            .env("CFG_VERSION", self.rust_version())
            .env("CFG_PREFIX", &self.config.install.prefix);

        let libdir_relative = self.config.libdir_relative().unwrap_or(Path::new("lib"));
        cargo.env("CFG_LIBDIR_RELATIVE", libdir_relative);

        // If we're not building a compiler with debugging information then remove
        // these two env vars which would be set otherwise.
        if self.config.rust.debuginfo_only_std() {
            cargo.env_remove("RUSTC_DEBUGINFO");
            cargo.env_remove("RUSTC_DEBUGINFO_LINES");
        }

        if let Some(ref ver_date) = self.rust_info.commit_date() {
            cargo.env("CFG_VER_DATE", ver_date);
        }
        if let Some(ref ver_hash) = self.rust_info.sha() {
            cargo.env("CFG_VER_HASH", ver_hash);
        }
        if !self.unstable_features() {
            cargo.env("CFG_DISABLE_UNSTABLE_FEATURES", "1");
        }
        if let Some(ref s) = self.config.rust.default_linker {
            cargo.env("CFG_DEFAULT_LINKER", s);
        }
        if self.config.rust.experimental_parallel_queries {
            cargo.env("RUSTC_PARALLEL_QUERIES", "1");
        }
    }

    fn tool_cargo(&self, target: Interned<String>, cargo: &mut Command) {
        // We don't want to build tools dynamically as they'll be running across
        // stages and such and it's just easier if they're not dynamically linked.
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        if let Some(dir) = self.openssl_install_dir(target) {
            cargo.env("OPENSSL_STATIC", "1");
            cargo.env("OPENSSL_DIR", dir);
            cargo.env("LIBZ_SYS_STATIC", "1");
        }

        // if tools are using lzma we want to force the build script to build its
        // own copy
        cargo.env("LZMA_API_STATIC", "1");
        cargo.env("CFG_RELEASE_CHANNEL", &self.config.rust.channel);
        cargo.env("CFG_VERSION", self.rust_version());
    }

    /// Prepares an invocation of `cargo` to be run.
    ///
    /// This will create a `Command` that represents a pending execution of
    /// Cargo. This cargo will be configured to use `compiler` as the actual
    /// rustc compiler, its output will be scoped by `mode`'s output directory,
    /// it will pass the `--target` flag for the specified `target`, and will be
    /// executing the Cargo command `cmd`.
    pub fn cargo(
        &self,
        compiler: Compiler,
        mode: Mode,
        target: Interned<String>,
        cmd: &'static str,
    ) -> CargoCommand {
        let mut cargo = Command::new(&self.config.general.initial_cargo);
        cargo.arg(cmd);
        let out_dir = self.stage_out(compiler, mode);
        self.clear_if_dirty(&out_dir, &self.rustc(compiler));

        match mode {
            Mode::Libstd => {
                self.std_cargo(compiler, target, &mut cargo);
            },
            Mode::Libtest => {
                self.test_cargo(&mut cargo);
                self.clear_if_dirty(&out_dir, &self.libstd_stamp(compiler, target));
            }
            Mode::Librustc => {
                self.rustc_cargo(&mut cargo);
                cargo
                    .arg("--features")
                    .arg(self.rustc_features())
                    .arg("--manifest-path")
                    .arg(self.config.src.join("src/rustc/Cargo.toml"));
                self.clear_if_dirty(&out_dir, &self.libstd_stamp(compiler, target));
                self.clear_if_dirty(&out_dir, &self.libtest_stamp(compiler, target));
            }
            Mode::CodegenBackend(backend) => {
                self.rustc_cargo(&mut cargo);
                let mut features = self.rustc_features().to_string();
                cargo
                    .arg("--manifest-path")
                    .arg(self.config.src.join("src/librustc_trans/Cargo.toml"));
                if backend == "emscripten" {
                    features.push_str(" emscripten");
                }
                cargo.arg("--features").arg(features);
                self.clear_if_dirty(&out_dir, &self.libstd_stamp(compiler, target));
                self.clear_if_dirty(&out_dir, &self.libtest_stamp(compiler, target));
            }
            Mode::TestTool => {
                self.tool_cargo(target, &mut cargo);
                self.clear_if_dirty(&out_dir, &self.libstd_stamp(compiler, target));
                self.clear_if_dirty(&out_dir, &self.libtest_stamp(compiler, target));
            }
            Mode::RustcTool => {
                self.tool_cargo(target, &mut cargo);
                self.clear_if_dirty(&out_dir, &self.libstd_stamp(compiler, target));
                self.clear_if_dirty(&out_dir, &self.libtest_stamp(compiler, target));
                self.clear_if_dirty(&out_dir, &self.librustc_stamp(compiler, target));
            }
        }

        cargo
            .env("CARGO_TARGET_DIR", out_dir)
            .arg("--target")
            .arg(target);

        // If we were invoked from `make` then that's already got a jobserver
        // set up for us so no need to tell Cargo about jobs all over again.
        if env::var_os("MAKEFLAGS").is_none() && env::var_os("MFLAGS").is_none() {
            cargo.arg("-j").arg(self.jobs().to_string());
        }

        // FIXME: Temporary fix for https://github.com/rust-lang/cargo/issues/3005
        // Force cargo to output binaries with disambiguating hashes in the name
        cargo.env("__CARGO_DEFAULT_LIB_METADATA", &self.config.rust.channel);

        let stage;
        if compiler.stage == 0 && self.config.general.local_rebuild {
            // Assume the local-rebuild rustc already has stage1 features.
            stage = 1;
        } else {
            stage = compiler.stage;
        }

        let mut extra_args = env::var(&format!("RUSTFLAGS_STAGE_{}", stage)).unwrap_or_default();
        if stage != 0 {
            let s = env::var("RUSTFLAGS_STAGE_NOT_0").unwrap_or_default();
            extra_args.push_str(" ");
            extra_args.push_str(&s);
        }

        if !extra_args.is_empty() {
            cargo.env(
                "RUSTFLAGS",
                format!(
                    "{} {}",
                    env::var("RUSTFLAGS").unwrap_or_default(),
                    extra_args
                ),
            );
        }

        // Customize the compiler we're running. Specify the compiler to cargo
        // as our shim and then pass it some various options used to configure
        // how the actual compiler itself is called.
        //
        // These variables are primarily all read by
        // src/bootstrap/bin/{rustc.rs,rustdoc.rs}
        cargo
            .env("RUSTBUILD_NATIVE_DIR", self.native_dir(target))
            .env(
                "RUSTC",
                self.config.general.out.join("bootstrap/debug/rustc"),
            )
            .env("RUSTC_REAL", self.rustc(compiler))
            .env("RUSTC_STAGE", stage.to_string())
            .env(
                "RUSTC_DEBUG_ASSERTIONS",
                self.config.rust.debug_assertions().to_string(),
            )
            .env("RUSTC_SYSROOT", self.sysroot(compiler))
            .env("RUSTC_LIBDIR", self.rustc_libdir(compiler))
            .env("RUSTC_RPATH", self.config.rust.rpath.to_string())
            .env(
                "RUSTDOC",
                self.config.general.out.join("bootstrap/debug/rustdoc"),
            )
            .env(
                "RUSTDOC_REAL",
                if cmd == "doc" || cmd == "test" {
                    self.rustdoc(compiler.host)
                } else {
                    PathBuf::from("/path/to/nowhere/rustdoc/not/required")
                },
            )
            .env("TEST_MIRI", self.config.rust.test_miri.to_string())
            .env("RUSTC_ERROR_METADATA_DST", self.extended_error_dir());

        if let Some(host_linker) = self.linker(compiler.host) {
            cargo.env("RUSTC_HOST_LINKER", host_linker);
        }
        if let Some(target_linker) = self.linker(target) {
            cargo.env("RUSTC_TARGET_LINKER", target_linker);
        }
        if let Some(ref error_format) = self.config.rustc_error_format {
            cargo.env("RUSTC_ERROR_FORMAT", error_format);
        }
        if cmd != "build" && cmd != "check" {
            cargo.env(
                "RUSTDOC_LIBDIR",
                self.rustc_libdir(self.compiler(2, self.config.general.build)),
            );
        }

        if mode != Mode::TestTool && mode != Mode::RustcTool {
            // Tools don't get debuginfo right now, e.g. cargo and rls don't
            // get compiled with debuginfo.
            // Adding debuginfo increases their sizes by a factor of 3-4.
            cargo.env("RUSTC_DEBUGINFO", self.config.rust.debuginfo().to_string());
            cargo.env(
                "RUSTC_DEBUGINFO_LINES",
                self.config.rust.debuginfo_lines().to_string(),
            );
            cargo.env("RUSTC_FORCE_UNSTABLE", "1");

            // Currently the compiler depends on crates from crates.io, and
            // then other crates can depend on the compiler (e.g. proc-macro
            // crates). Let's say, for example that rustc itself depends on the
            // bitflags crate. If an external crate then depends on the
            // bitflags crate as well, we need to make sure they don't
            // conflict, even if they pick the same version of bitflags. We'll
            // want to make sure that e.g. a plugin and rustc each get their
            // own copy of bitflags.

            // Cargo ensures that this works in general through the -C metadata
            // flag. This flag will frob the symbols in the binary to make sure
            // they're different, even though the source code is the exact
            // same. To solve this problem for the compiler we extend Cargo's
            // already-passed -C metadata flag with our own. Our rustc.rs
            // wrapper around the actual rustc will detect -C metadata being
            // passed and frob it with this extra string we're passing in.
            cargo.env("RUSTC_METADATA_SUFFIX", "rustc");
        }

        if let Some(x) = self.crt_static(target) {
            cargo.env("RUSTC_CRT_STATIC", x.to_string());
        }

        // Enable usage of unstable features
        cargo.env("RUSTC_BOOTSTRAP", "1");
        self.add_rust_test_threads(&mut cargo);

        // Almost all of the crates that we compile as part of the bootstrap may
        // have a build script, including the standard library. To compile a
        // build script, however, it itself needs a standard library! This
        // introduces a bit of a pickle when we're compiling the standard
        // library itself.
        //
        // To work around this we actually end up using the snapshot compiler
        // (stage0) for compiling build scripts of the standard library itself.
        // The stage0 compiler is guaranteed to have a libstd available for use.
        //
        // For other crates, however, we know that we've already got a standard
        // library up and running, so we can use the normal compiler to compile
        // build scripts in that situation.
        //
        // If LLVM support is disabled we need to use the snapshot compiler to compile
        // build scripts, as the new compiler doesn't support executables.
        if mode == Mode::Libstd || !self.config.llvm.enabled {
            cargo
                .env("RUSTC_SNAPSHOT", &self.config.general.initial_rustc)
                .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_snapshot_libdir());
        } else {
            self.ensure(compile::Std {
                compiler,
                target: compiler.host,
            });
            cargo
                .env("RUSTC_SNAPSHOT", self.rustc(compiler))
                .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_libdir(compiler));
        }

        // Ignore incremental modes except for stage0, since we're
        // not guaranteeing correctness across builds if the compiler
        // is changing under your feet.`
        if self.config.incremental && compiler.stage == 0 {
            cargo.env("CARGO_INCREMENTAL", "1");
        }

        if let Some(ref on_fail) = self.config.on_fail {
            cargo.env("RUSTC_ON_FAIL", on_fail);
        }

        cargo.env("RUSTC_VERBOSE", format!("{}", self.config.general.verbose));

        // Throughout the build Cargo can execute a number of build scripts
        // compiling C/C++ code and we need to pass compilers, archivers, flags, etc
        // obtained previously to those build scripts.
        // Build scripts use either the `cc` crate or `configure/make` so we pass
        // the options through environment variables that are fetched and understood by both.
        //
        // FIXME: the guard against msvc shouldn't need to be here
        if !target.contains("msvc") {
            let cc = self.cc(target);
            cargo.env(format!("CC_{}", target), cc).env("CC", cc);

            let cflags = self.cflags(target).join(" ");
            cargo
                .env(format!("CFLAGS_{}", target), cflags.clone())
                .env("CFLAGS", cflags.clone());

            if let Some(ar) = self.ar(target) {
                let ranlib = format!("{} s", ar.display());
                cargo
                    .env(format!("AR_{}", target), ar)
                    .env("AR", ar)
                    .env(format!("RANLIB_{}", target), ranlib.clone())
                    .env("RANLIB", ranlib);
            }

            if let Ok(cxx) = self.cxx(target) {
                cargo
                    .env(format!("CXX_{}", target), cxx)
                    .env("CXX", cxx)
                    .env(format!("CXXFLAGS_{}", target), cflags.clone())
                    .env("CXXFLAGS", cflags);
            }
        }

        if mode == Mode::Libstd && self.config.general.extended && compiler.is_final_stage(self) {
            cargo.env("RUSTC_SAVE_ANALYSIS", "api".to_string());
        }

        // For `cargo doc` invocations, make rustdoc print the Rust version into the docs
        cargo.env("RUSTDOC_CRATE_VERSION", self.rust_version());

        // Environment variables *required* throughout the build
        //
        // FIXME: should update code to not require this env var
        cargo.env("CFG_COMPILER_HOST_TRIPLE", target);

        // Set this for all builds to make sure doc builds also get it.
        cargo.env("CFG_RELEASE_CHANNEL", &self.config.rust.channel);

        // This one's a bit tricky. As of the time of this writing the compiler
        // links to the `winapi` crate on crates.io. This crate provides raw
        // bindings to Windows system functions, sort of like libc does for
        // Unix. This crate also, however, provides "import libraries" for the
        // MinGW targets. There's an import library per dll in the windows
        // distribution which is what's linked to. These custom import libraries
        // are used because the winapi crate can reference Windows functions not
        // present in the MinGW import libraries.
        //
        // For example MinGW may ship libdbghelp.a, but it may not have
        // references to all the functions in the dbghelp dll. Instead the
        // custom import library for dbghelp in the winapi crates has all this
        // information.
        //
        // Unfortunately for us though the import libraries are linked by
        // default via `-ldylib=winapi_foo`. That is, they're linked with the
        // `dylib` type with a `winapi_` prefix (so the winapi ones don't
        // conflict with the system MinGW ones). This consequently means that
        // the binaries we ship of things like rustc_trans (aka the rustc_trans
        // DLL) when linked against *again*, for example with procedural macros
        // or plugins, will trigger the propagation logic of `-ldylib`, passing
        // `-lwinapi_foo` to the linker again. This isn't actually available in
        // our distribution, however, so the link fails.
        //
        // To solve this problem we tell winapi to not use its bundled import
        // libraries. This means that it will link to the system MinGW import
        // libraries by default, and the `-ldylib=foo` directives will still get
        // passed to the final linker, but they'll look like `-lfoo` which can
        // be resolved because MinGW has the import library. The downside is we
        // don't get newer functions from Windows, but we don't use any of them
        // anyway.
        cargo.env("WINAPI_NO_BUNDLED_LIBRARIES", "1");

        if self.is_very_verbose() {
            cargo.arg("-v");
        }

        // This must be kept before the thinlto check, as we set codegen units
        // to 1 forcibly there.
        if let Some(n) = self.config.rust.codegen_units() {
            cargo.env("RUSTC_CODEGEN_UNITS", n.to_string());
        }

        if self.config.rust.optimize() {
            // FIXME: cargo bench does not accept `--release`
            if cmd != "bench" {
                cargo.arg("--release");
            }

            if self.config.rust.codegen_units().is_none() && self.is_rust_llvm(compiler.host)
                && self.config.rust.thinlto
            {
                cargo.env("RUSTC_THINLTO", "1");
            } else if self.config.rust.codegen_units().is_none() {
                // Generally, if ThinLTO has been disabled for some reason, we
                // want to set the codegen units to 1. However, we shouldn't do
                // this if the option was specifically set by the user.
                cargo.env("RUSTC_CODEGEN_UNITS", "1");
            }
        }

        if self.config.general.locked_deps {
            cargo.arg("--locked");
        }
        if self.config.general.vendor || self.config.is_sudo {
            cargo.arg("--frozen");
        }

        self.ci_env.force_coloring_in_ci(&mut cargo);

        CargoCommand {
            cargo,
            mode,
            target,
            cmd,
            compiler,
            builder: self,
        }
    }

    /// Ensure that a given step is built, returning it's output. This will
    /// cache the step, so it is safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are built.
    pub fn ensure<S: Step>(&'a self, step: S) -> S::Output {
        {
            let mut stack = self.stack.borrow_mut();
            for stack_step in stack.iter() {
                // should skip
                if stack_step
                    .downcast_ref::<S>()
                    .map_or(true, |stack_step| *stack_step != step)
                {
                    continue;
                }
                let mut out = String::new();
                out += &format!("\n\nCycle in build detected when adding {:?}\n", step);
                for el in stack.iter().rev() {
                    out += &format!("\t{:?}\n", el);
                }
                panic!(out);
            }
            if let Some(out) = self.cache.get(&step) {
                self.verbose(&format!("{}c {:?}", "  ".repeat(stack.len()), step));

                return out;
            }
            self.verbose(&format!("{}> {:?}", "  ".repeat(stack.len()), step));
            stack.push(Box::new(step.clone()));
        }
        let out = step.clone().run(self);
        {
            let mut stack = self.stack.borrow_mut();
            let cur_step = stack.pop().expect("step stack empty");
            assert_eq!(cur_step.downcast_ref(), Some(&step));
        }
        self.verbose(&format!(
            "{}< {:?}",
            "  ".repeat(self.stack.borrow().len()),
            step
        ));
        self.cache.put(step, out.clone());
        out
    }
}

pub struct CargoCommand<'a> {
    builder: &'a Builder<'a>,
    cargo: Command,
    mode: Mode,
    compiler: Compiler,
    target: Interned<String>,
    cmd: &'static str,
}

impl<'a> Deref for CargoCommand<'a> {
    type Target = Command;
    fn deref(&self) -> &Command {
        &self.cargo
    }
}

impl<'a> DerefMut for CargoCommand<'a> {
    fn deref_mut(&mut self) -> &mut Command {
        &mut self.cargo
    }
}

impl<'a> CargoCommand<'a> {
    pub fn run(&mut self) {
        let stamp = match self.mode {
            Mode::Libstd => self.builder.libstd_stamp(self.compiler, self.target),
            Mode::Libtest => self.builder.libtest_stamp(self.compiler, self.target),
            Mode::Librustc => self.builder.librustc_stamp(self.compiler, self.target),
            Mode::CodegenBackend(backend) => {
                self.builder.codegen_backend_stamp(self.compiler, self.target, &*backend)
            }
            Mode::TestTool | Mode::RustcTool => {
                panic!("did not expect to execute with tools");
            }
        };

        let is_check = self.cmd == "check";

        // Instruct Cargo to give us json messages on stdout, critically leaving
        // stderr as piped so we can get those pretty colors.
        self.cargo
            .arg("--message-format")
            .arg("json")
            .stdout(Stdio::piped());

        if stderr_isatty() && self.builder.ci_env == CiEnv::None {
            // since we pass message-format=json to cargo, we need to tell the rustc
            // wrapper to give us colored output if necessary. This is because we
            // only want Cargo's JSON output, not rustcs.
            self.cargo.env("RUSTC_COLOR", "1");
        }

        self.builder.verbose(&format!("running: {:?}", self.cargo));
        let mut child = match self.cargo.spawn() {
            Ok(child) => child,
            Err(e) => panic!("failed to execute command: {:?}\nerror: {}", self.cargo, e),
        };

        // `target_root_dir` looks like $dir/$target/release
        let target_root_dir = stamp.parent().unwrap();
        // `target_deps_dir` looks like $dir/$target/release/deps
        let target_deps_dir = target_root_dir.join("deps");
        // `host_root_dir` looks like $dir/release
        let host_root_dir = target_root_dir.parent().unwrap() // chop off `release`
                                        .parent().unwrap() // chop off `$target`
                                        .join(target_root_dir.file_name().unwrap());

        // Spawn Cargo slurping up its JSON output. We'll start building up the
        // `deps` array of all files it generated along with a `toplevel` array of
        // files we need to probe for later.
        let mut deps = Vec::new();
        let mut toplevel = Vec::new();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        for line in stdout.lines() {
            let line = t!(line);
            let json: serde_json::Value = if line.starts_with("{") {
                t!(serde_json::from_str(&line))
            } else {
                // If this was informational, just print it out and continue
                println!("{}", line);
                continue;
            };
            if json["reason"].as_str() != Some("compiler-artifact") {
                continue;
            }
            for filename in json["filenames"].as_array().unwrap() {
                let filename = filename.as_str().unwrap();
                // Skip files like executables
                if !filename.ends_with(".rlib") && !filename.ends_with(".lib") && !is_dylib(&filename)
                    && !(is_check && filename.ends_with(".rmeta"))
                {
                    continue;
                }

                let filename = Path::new(filename);

                // If this was an output file in the "host dir" we don't actually
                // worry about it, it's not relevant for us.
                if filename.starts_with(&host_root_dir) {
                    continue;
                }

                // If this was output in the `deps` dir then this is a precise file
                // name (hash included) so we start tracking it.
                if filename.starts_with(&target_deps_dir) {
                    deps.push(filename.to_path_buf());
                    continue;
                }

                // Otherwise this was a "top level artifact" which right now doesn't
                // have a hash in the name, but there's a version of this file in
                // the `deps` folder which *does* have a hash in the name. That's
                // the one we'll want to we'll probe for it later.
                //
                // We do not use `Path::file_stem` or `Path::extension` here,
                // because some generated files may have multiple extensions e.g.
                // `std-<hash>.dll.lib` on Windows. The aforementioned methods only
                // split the file name by the last extension (`.lib`) while we need
                // to split by all extensions (`.dll.lib`).
                let expected_len = t!(filename.metadata()).len();
                let filename = filename.file_name().unwrap().to_str().unwrap();
                let mut parts = filename.splitn(2, '.');
                let file_stem = parts.next().unwrap().to_owned();
                let extension = parts.next().unwrap().to_owned();

                toplevel.push((file_stem, extension, expected_len));
            }
        }

        // Make sure Cargo actually succeeded after we read all of its stdout.
        let status = t!(child.wait());
        if !status.success() {
            panic!(
                "command did not execute successfully: {:?}\n\
                expected success, got: {}",
                self.cargo, status
            );
        }

        // Ok now we need to actually find all the files listed in `toplevel`. We've
        // got a list of prefix/extensions and we basically just need to find the
        // most recent file in the `deps` folder corresponding to each one.
        let contents = t!(target_deps_dir.read_dir())
            .map(|e| t!(e))
            .map(|e| {
                (
                    e.path(),
                    e.file_name().into_string().unwrap(),
                    t!(e.metadata()),
                )
            })
            .collect::<Vec<_>>();
        for (prefix, extension, expected_len) in toplevel {
            let candidates = contents.iter().filter(|&&(_, ref filename, ref meta)| {
                filename.starts_with(&prefix[..]) && filename[prefix.len()..].starts_with("-")
                    && filename.ends_with(&extension[..]) && meta.len() == expected_len
            });
            let max = candidates
                .max_by_key(|&&(_, _, ref metadata)| FileTime::from_last_modification_time(metadata));
            let path_to_add = match max {
                Some(triple) => triple.0.to_str().unwrap(),
                None => panic!("no output generated for {:?} {:?}", prefix, extension),
            };
            if is_dylib(path_to_add) {
                let candidate = format!("{}.lib", path_to_add);
                let candidate = PathBuf::from(candidate);
                if candidate.exists() {
                    deps.push(candidate);
                }
            }
            deps.push(path_to_add.into());
        }

        // Now we want to update the contents of the stamp file, if necessary. First
        // we read off the previous contents along with its mtime. If our new
        // contents (the list of files to copy) is different or if any dep's mtime
        // is newer then we rewrite the stamp file.
        deps.sort();
        let mut stamp_contents = Vec::new();
        if let Ok(mut f) = File::open(&stamp) {
            t!(f.read_to_end(&mut stamp_contents));
        }
        let stamp_mtime = mtime(&stamp);
        let mut new_contents = Vec::new();
        let mut max = None;
        let mut max_path = None;
        for dep in deps.iter() {
            let mtime = mtime(dep);
            if Some(mtime) > max {
                max = Some(mtime);
                max_path = Some(dep.clone());
            }
            new_contents.extend(dep.to_str().unwrap().as_bytes());
            new_contents.extend(b"\0");
        }
        let max = max.unwrap();
        let max_path = max_path.unwrap();
        if stamp_contents == new_contents && max <= stamp_mtime {
            self.builder.verbose(&format!(
                "not updating {:?}; contents equal and {} <= {}",
                stamp, max, stamp_mtime
            ));
            return;
        }
        if max > stamp_mtime {
            self.builder.verbose(&format!("updating {:?} as {:?} changed", stamp, max_path));
        } else {
            self.builder.verbose(&format!("updating {:?} as deps changed", stamp));
        }
        t!(t!(File::create(&stamp)).write_all(&new_contents));
    }

}

// Avoiding a dependency on winapi to keep compile times down
#[cfg(unix)]
fn stderr_isatty() -> bool {
    use libc;
    unsafe { libc::isatty(libc::STDERR_FILENO) != 0 }
}
#[cfg(windows)]
fn stderr_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut DWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_ERROR_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}
