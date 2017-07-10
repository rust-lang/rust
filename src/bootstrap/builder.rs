// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;
use std::hash::Hash;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::ops::Deref;
use std::any::{TypeId, Any};

use compile;
use install;
use dist;
use util::{exe, libdir, add_lib_path};
use {Build, Mode};
use cache::{INTERNER, Interned, Cache};
use check;
use flags::Subcommand;
use doc;
use tool;

pub use Compiler;

pub struct Builder<'a> {
    pub build: &'a Build,
    pub top_stage: u32,
    pub kind: Kind,
    cache: Cache,
    stack: RefCell<Vec<(TypeId, Box<Any>)>>,
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

    /// Run this rule for all targets, but only with the native host.
    const ONLY_BUILD_TARGETS: bool = false;

    /// Only run this step with the build triple as host and target.
    const ONLY_BUILD: bool = false;

    /// Primary function to execute this rule. Can call `builder.ensure(...)`
    /// with other steps to run those.
    fn run(self, builder: &Builder) -> Self::Output;

    /// When bootstrap is passed a set of paths, this controls whether this rule
    /// will execute. However, it does not get called in a "default" context
    /// when we are not passed any paths; in that case, make_run is called
    /// directly.
    fn should_run(builder: &Builder, path: &Path) -> bool;

    /// Build up a "root" rule, either as a default rule or from a path passed
    /// to us.
    ///
    /// When path is `None`, we are executing in a context where no paths were
    /// passed. When `./x.py build` is run, for example, this rule could get
    /// called if it is in the correct list below with a path of `None`.
    fn make_run(
        _builder: &Builder,
        _path: Option<&Path>,
        _host: Interned<String>,
        _target: Interned<String>,
    ) {
        // It is reasonable to not have an implementation of make_run for rules
        // who do not want to get called from the root context. This means that
        // they are likely dependencies (e.g., sysroot creation) or similar, and
        // as such calling them from ./x.py isn't logical.
        unimplemented!()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Kind {
    Build,
    Test,
    Bench,
    Dist,
    Doc,
    Install,
}

macro_rules! check {
    ($self:ident, $paths:ident, $($rule:ty),+ $(,)*) => {{
        let paths = $paths;
        if paths.is_empty() {
            $({
                if <$rule>::DEFAULT {
                    $self.maybe_run::<$rule>(None);
                }
            })+
        } else {
            for path in paths {
                let mut attempted_run = false;
                $({
                    if <$rule>::should_run($self, path) {
                        attempted_run = true;
                        $self.maybe_run::<$rule>(Some(path));
                    }
                })+
                if !attempted_run {
                    eprintln!("Warning: no rules matched {}.", path.display());
                }
            }
        }
    }};
}

impl<'a> Builder<'a> {
    pub fn run(build: &Build) {
        let (kind, paths) = match build.flags.cmd {
            Subcommand::Build { ref paths } => (Kind::Build, &paths[..]),
            Subcommand::Doc { ref paths } => (Kind::Doc, &paths[..]),
            Subcommand::Test { ref paths, .. } => (Kind::Test, &paths[..]),
            Subcommand::Bench { ref paths, .. } => (Kind::Bench, &paths[..]),
            Subcommand::Dist { ref paths } => (Kind::Dist, &paths[..]),
            Subcommand::Install { ref paths } => (Kind::Install, &paths[..]),
            Subcommand::Clean => panic!(),
        };

        let builder = Builder {
            build: build,
            top_stage: build.flags.stage.unwrap_or(2),
            kind: kind,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
        };

        let builder = &builder;
        match builder.kind {
            Kind::Build => check!(builder, paths, compile::Std, compile::Test, compile::Rustc,
                compile::StartupObjects, tool::BuildManifest, tool::Rustbook, tool::ErrorIndex,
                tool::UnstableBookGen, tool::Tidy, tool::Linkchecker, tool::CargoTest,
                tool::Compiletest, tool::RemoteTestServer, tool::RemoteTestClient,
                tool::RustInstaller, tool::Cargo, tool::Rls),
            Kind::Test => check!(builder, paths, check::Tidy, check::Bootstrap, check::Compiletest,
                check::Krate, check::KrateLibrustc, check::Linkcheck, check::Cargotest,
                check::Cargo, check::Docs, check::ErrorIndex, check::Distcheck),
            Kind::Bench => check!(builder, paths, check::Krate, check::KrateLibrustc),
            Kind::Doc => builder.default_doc(Some(paths)),
            Kind::Dist => check!(builder, paths, dist::Docs, dist::Mingw, dist::Rustc,
                dist::DebuggerScripts, dist::Std, dist::Analysis, dist::Src,
                dist::PlainSourceTarball, dist::Cargo, dist::Rls, dist::Extended, dist::HashSign),
            Kind::Install => check!(builder, paths, install::Docs, install::Std, install::Cargo,
                install::Rls, install::Analysis, install::Src, install::Rustc),
        }
    }

    pub fn default_doc(&self, paths: Option<&[PathBuf]>) {
        let paths = paths.unwrap_or(&[]);
        check!(self, paths, doc::UnstableBook, doc::UnstableBookGen, doc::Rustbook, doc::TheBook,
            doc::Standalone, doc::Std, doc::Test, doc::Rustc, doc::ErrorIndex,
            doc::Nomicon, doc::Reference);
    }

    /// Obtain a compiler at a given stage and for a given host. Explictly does
    /// not take `Compiler` since all `Compiler` instances are meant to be
    /// obtained through this function, since it ensures that they are valid
    /// (i.e., built and assembled).
    pub fn compiler(&self, stage: u32, host: Interned<String>) -> Compiler {
        self.ensure(compile::Assemble { target_compiler: Compiler { stage, host } })
    }

    pub fn sysroot(&self, compiler: Compiler) -> Interned<PathBuf> {
        self.ensure(compile::Sysroot { compiler })
    }

    /// Returns the libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
    pub fn sysroot_libdir(
        &self, compiler: Compiler, target: Interned<String>
    ) -> Interned<PathBuf> {
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        struct Libdir {
            compiler: Compiler,
            target: Interned<String>,
        }
        impl Step for Libdir {
            type Output = Interned<PathBuf>;

            fn should_run(_builder: &Builder, _path: &Path) -> bool {
                false
            }

            fn run(self, builder: &Builder) -> Interned<PathBuf> {
                let compiler = self.compiler;
                let lib = if compiler.stage >= 2 && builder.build.config.libdir_relative.is_some() {
                    builder.build.config.libdir_relative.clone().unwrap()
                } else {
                    PathBuf::from("lib")
                };
                let sysroot = builder.sysroot(self.compiler).join(lib)
                    .join("rustlib").join(self.target).join("lib");
                let _ = fs::remove_dir_all(&sysroot);
                t!(fs::create_dir_all(&sysroot));
                INTERNER.intern_path(sysroot)
            }
        }
        self.ensure(Libdir { compiler, target })
    }

    /// Returns the compiler's libdir where it stores the dynamic libraries that
    /// it itself links against.
    ///
    /// For example this returns `<sysroot>/lib` on Unix and `<sysroot>/bin` on
    /// Windows.
    pub fn rustc_libdir(&self, compiler: Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.build.rustc_snapshot_libdir()
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
            return
        }

        add_lib_path(vec![self.rustc_libdir(compiler)], cmd);
    }

    /// Get a path to the compiler specified.
    pub fn rustc(&self, compiler: Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.initial_rustc.clone()
        } else {
            self.sysroot(compiler).join("bin").join(exe("rustc", &compiler.host))
        }
    }

    /// Get the `rustdoc` executable next to the specified compiler
    pub fn rustdoc(&self, compiler: Compiler) -> PathBuf {
        let mut rustdoc = self.rustc(compiler);
        rustdoc.pop();
        rustdoc.push(exe("rustdoc", &compiler.host));
        rustdoc
    }

    /// Prepares an invocation of `cargo` to be run.
    ///
    /// This will create a `Command` that represents a pending execution of
    /// Cargo. This cargo will be configured to use `compiler` as the actual
    /// rustc compiler, its output will be scoped by `mode`'s output directory,
    /// it will pass the `--target` flag for the specified `target`, and will be
    /// executing the Cargo command `cmd`.
    pub fn cargo(&self,
             compiler: Compiler,
             mode: Mode,
             target: Interned<String>,
             cmd: &str) -> Command {
        let mut cargo = Command::new(&self.initial_cargo);
        let out_dir = self.stage_out(compiler, mode);
        cargo.env("CARGO_TARGET_DIR", out_dir)
             .arg(cmd)
             .arg("-j").arg(self.jobs().to_string())
             .arg("--target").arg(target);

        // FIXME: Temporary fix for https://github.com/rust-lang/cargo/issues/3005
        // Force cargo to output binaries with disambiguating hashes in the name
        cargo.env("__CARGO_DEFAULT_LIB_METADATA", &self.config.channel);

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
        cargo.env("RUSTBUILD_NATIVE_DIR", self.native_dir(target))
             .env("RUSTC", self.out.join("bootstrap/debug/rustc"))
             .env("RUSTC_REAL", self.rustc(compiler))
             .env("RUSTC_STAGE", stage.to_string())
             .env("RUSTC_CODEGEN_UNITS",
                  self.config.rust_codegen_units.to_string())
             .env("RUSTC_DEBUG_ASSERTIONS",
                  self.config.rust_debug_assertions.to_string())
             .env("RUSTC_SYSROOT", self.sysroot(compiler))
             .env("RUSTC_LIBDIR", self.rustc_libdir(compiler))
             .env("RUSTC_RPATH", self.config.rust_rpath.to_string())
             .env("RUSTDOC", self.out.join("bootstrap/debug/rustdoc"))
             .env("RUSTDOC_REAL", self.rustdoc(compiler))
             .env("RUSTC_FLAGS", self.rustc_flags(target).join(" "));

        if mode != Mode::Tool {
            // Tools don't get debuginfo right now, e.g. cargo and rls don't
            // get compiled with debuginfo.
            cargo.env("RUSTC_DEBUGINFO", self.config.rust_debuginfo.to_string())
                 .env("RUSTC_DEBUGINFO_LINES", self.config.rust_debuginfo_lines.to_string())
                 .env("RUSTC_FORCE_UNSTABLE", "1");

            // Currently the compiler depends on crates from crates.io, and
            // then other crates can depend on the compiler (e.g. proc-macro
            // crates). Let's say, for example that rustc itself depends on the
            // bitflags crate. If an external crate then depends on the
            // bitflags crate as well, we need to make sure they don't
            // conflict, even if they pick the same verison of bitflags. We'll
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
        if mode == Mode::Libstd {
            cargo.env("RUSTC_SNAPSHOT", &self.initial_rustc)
                 .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_snapshot_libdir());
        } else {
            cargo.env("RUSTC_SNAPSHOT", self.rustc(compiler))
                 .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_libdir(compiler));
        }

        // Ignore incremental modes except for stage0, since we're
        // not guaranteeing correctness across builds if the compiler
        // is changing under your feet.`
        if self.flags.incremental && compiler.stage == 0 {
            let incr_dir = self.incremental_dir(compiler);
            cargo.env("RUSTC_INCREMENTAL", incr_dir);
        }

        if let Some(ref on_fail) = self.flags.on_fail {
            cargo.env("RUSTC_ON_FAIL", on_fail);
        }

        cargo.env("RUSTC_VERBOSE", format!("{}", self.verbosity));

        // Specify some various options for build scripts used throughout
        // the build.
        //
        // FIXME: the guard against msvc shouldn't need to be here
        if !target.contains("msvc") {
            cargo.env(format!("CC_{}", target), self.cc(target))
                 .env(format!("AR_{}", target), self.ar(target).unwrap()) // only msvc is None
                 .env(format!("CFLAGS_{}", target), self.cflags(target).join(" "));

            if let Ok(cxx) = self.cxx(target) {
                 cargo.env(format!("CXX_{}", target), cxx);
            }
        }

        if mode == Mode::Libstd && self.config.extended && compiler.is_final_stage(self) {
            cargo.env("RUSTC_SAVE_ANALYSIS", "api".to_string());
        }

        // When being built Cargo will at some point call `nmake.exe` on Windows
        // MSVC. Unfortunately `nmake` will read these two environment variables
        // below and try to intepret them. We're likely being run, however, from
        // MSYS `make` which uses the same variables.
        //
        // As a result, to prevent confusion and errors, we remove these
        // variables from our environment to prevent passing MSYS make flags to
        // nmake, causing it to blow up.
        if cfg!(target_env = "msvc") {
            cargo.env_remove("MAKE");
            cargo.env_remove("MAKEFLAGS");
        }

        // Environment variables *required* throughout the build
        //
        // FIXME: should update code to not require this env var
        cargo.env("CFG_COMPILER_HOST_TRIPLE", target);

        if self.is_verbose() {
            cargo.arg("-v");
        }
        // FIXME: cargo bench does not accept `--release`
        if self.config.rust_optimize && cmd != "bench" {
            cargo.arg("--release");
        }
        if self.config.locked_deps {
            cargo.arg("--locked");
        }
        if self.config.vendor || self.is_sudo {
            cargo.arg("--frozen");
        }

        self.ci_env.force_coloring_in_ci(&mut cargo);

        cargo
    }

    fn maybe_run<S: Step>(&self, path: Option<&Path>) {
        let build = self.build;
        let hosts = if S::ONLY_BUILD_TARGETS || S::ONLY_BUILD {
            &build.config.host[..1]
        } else {
            &build.hosts
        };

        // Determine the actual targets participating in this rule.
        // NOTE: We should keep the full projection from build triple to
        // the hosts for the dist steps, now that the hosts array above is
        // truncated to avoid duplication of work in that case. Therefore
        // the original non-shadowed hosts array is used below.
        let targets = if S::ONLY_HOSTS {
            // If --target was specified but --host wasn't specified,
            // don't run any host-only tests. Also, respect any `--host`
            // overrides as done for `hosts`.
            if build.flags.host.len() > 0 {
                &build.flags.host[..]
            } else if build.flags.target.len() > 0 {
                &[]
            } else if S::ONLY_BUILD {
                &build.config.host[..1]
            } else {
                &build.config.host[..]
            }
        } else {
            &build.targets
        };

        for host in hosts {
            for target in targets {
                S::make_run(self, path, *host, *target);
            }
        }
    }

    /// Ensure that a given step is built, returning it's output. This will
    /// cache the step, so it is safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are built.
    pub fn ensure<S: Step>(&'a self, step: S) -> S::Output {
        let type_id = TypeId::of::<S>();
        {
            let mut stack = self.stack.borrow_mut();
            for &(stack_type_id, ref stack_step) in stack.iter() {
                if !(type_id == stack_type_id && step == *stack_step.downcast_ref().unwrap()) {
                    continue
                }
                let mut out = String::new();
                out += &format!("\n\nCycle in build detected when adding {:?}\n", step);
                for el in stack.iter().rev() {
                    out += &format!("\t{:?}\n", el);
                }
                panic!(out);
            }
            if let Some(out) = self.cache.get(&step) {
                self.build.verbose(&format!("{}c {:?}", "  ".repeat(stack.len()), step));

                return out;
            }
            self.build.verbose(&format!("{}> {:?}", "  ".repeat(stack.len()), step));
            stack.push((type_id, Box::new(step.clone())));
        }
        let out = step.clone().run(self);
        {
            let mut stack = self.stack.borrow_mut();
            let (cur_type_id, cur_step) = stack.pop().expect("step stack empty");
            assert_eq!((cur_type_id, cur_step.downcast_ref()), (type_id, Some(&step)));
        }
        self.build.verbose(&format!("{}< {:?}", "  ".repeat(self.stack.borrow().len()), step));
        self.cache.put(step, out.clone());
        out
    }
}
