use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::env;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs;
use std::hash::Hash;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use build_helper::{output, t};

use crate::cache::{Cache, Interned, INTERNER};
use crate::check;
use crate::compile;
use crate::config::TargetSelection;
use crate::dist;
use crate::doc;
use crate::flags::{Color, Subcommand};
use crate::install;
use crate::native;
use crate::run;
use crate::test;
use crate::tool::{self, SourceType};
use crate::util::{self, add_dylib_path, add_link_lib_path, exe, libdir};
use crate::{Build, DocTests, GitRepo, Mode};

pub use crate::Compiler;
// FIXME: replace with std::lazy after it gets stabilized and reaches beta
use once_cell::sync::Lazy;

pub struct Builder<'a> {
    pub build: &'a Build,
    pub top_stage: u32,
    pub kind: Kind,
    cache: Cache,
    stack: RefCell<Vec<Box<dyn Any>>>,
    time_spent_on_dependencies: Cell<Duration>,
    pub paths: Vec<PathBuf>,
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

    /// Whether this step is run by default as part of its respective phase.
    /// `true` here can still be overwritten by `should_run` calling `default_condition`.
    const DEFAULT: bool = false;

    /// If true, then this rule should be skipped if --target was specified, but --host was not
    const ONLY_HOSTS: bool = false;

    /// Primary function to execute this rule. Can call `builder.ensure()`
    /// with other steps to run those.
    fn run(self, builder: &Builder<'_>) -> Self::Output;

    /// When bootstrap is passed a set of paths, this controls whether this rule
    /// will execute. However, it does not get called in a "default" context
    /// when we are not passed any paths; in that case, `make_run` is called
    /// directly.
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_>;

    /// Builds up a "root" rule, either as a default rule or from a path passed
    /// to us.
    ///
    /// When path is `None`, we are executing in a context where no paths were
    /// passed. When `./x.py build` is run, for example, this rule could get
    /// called if it is in the correct list below with a path of `None`.
    fn make_run(_run: RunConfig<'_>) {
        // It is reasonable to not have an implementation of make_run for rules
        // who do not want to get called from the root context. This means that
        // they are likely dependencies (e.g., sysroot creation) or similar, and
        // as such calling them from ./x.py isn't logical.
        unimplemented!()
    }
}

pub struct RunConfig<'a> {
    pub builder: &'a Builder<'a>,
    pub target: TargetSelection,
    pub path: PathBuf,
}

impl RunConfig<'_> {
    pub fn build_triple(&self) -> TargetSelection {
        self.builder.build.build
    }
}

struct StepDescription {
    default: bool,
    only_hosts: bool,
    should_run: fn(ShouldRun<'_>) -> ShouldRun<'_>,
    make_run: fn(RunConfig<'_>),
    name: &'static str,
}

/// Collection of paths used to match a task rule.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum PathSet {
    /// A collection of individual paths.
    ///
    /// These are generally matched as a path suffix. For example, a
    /// command-line value of `libstd` will match if `src/libstd` is in the
    /// set.
    Set(BTreeSet<PathBuf>),
    /// A "suite" of paths.
    ///
    /// These can match as a path suffix (like `Set`), or as a prefix. For
    /// example, a command-line value of `src/test/ui/abi/variadic-ffi.rs`
    /// will match `src/test/ui`. A command-line value of `ui` would also
    /// match `src/test/ui`.
    Suite(PathBuf),
}

impl PathSet {
    fn empty() -> PathSet {
        PathSet::Set(BTreeSet::new())
    }

    fn one<P: Into<PathBuf>>(path: P) -> PathSet {
        let mut set = BTreeSet::new();
        set.insert(path.into());
        PathSet::Set(set)
    }

    fn has(&self, needle: &Path) -> bool {
        match self {
            PathSet::Set(set) => set.iter().any(|p| p.ends_with(needle)),
            PathSet::Suite(suite) => suite.ends_with(needle),
        }
    }

    fn path(&self, builder: &Builder<'_>) -> PathBuf {
        match self {
            PathSet::Set(set) => set.iter().next().unwrap_or(&builder.build.src).to_path_buf(),
            PathSet::Suite(path) => PathBuf::from(path),
        }
    }
}

impl StepDescription {
    fn from<S: Step>() -> StepDescription {
        StepDescription {
            default: S::DEFAULT,
            only_hosts: S::ONLY_HOSTS,
            should_run: S::should_run,
            make_run: S::make_run,
            name: std::any::type_name::<S>(),
        }
    }

    fn maybe_run(&self, builder: &Builder<'_>, pathset: &PathSet) {
        if self.is_excluded(builder, pathset) {
            return;
        }

        // Determine the targets participating in this rule.
        let targets = if self.only_hosts { &builder.hosts } else { &builder.targets };

        for target in targets {
            let run = RunConfig { builder, path: pathset.path(builder), target: *target };
            (self.make_run)(run);
        }
    }

    fn is_excluded(&self, builder: &Builder<'_>, pathset: &PathSet) -> bool {
        if builder.config.exclude.iter().any(|e| pathset.has(e)) {
            eprintln!("Skipping {:?} because it is excluded", pathset);
            return true;
        }

        if !builder.config.exclude.is_empty() {
            eprintln!(
                "{:?} not skipped for {:?} -- not in {:?}",
                pathset, self.name, builder.config.exclude
            );
        }
        false
    }

    fn run(v: &[StepDescription], builder: &Builder<'_>, paths: &[PathBuf]) {
        let should_runs =
            v.iter().map(|desc| (desc.should_run)(ShouldRun::new(builder))).collect::<Vec<_>>();

        // sanity checks on rules
        for (desc, should_run) in v.iter().zip(&should_runs) {
            assert!(
                !should_run.paths.is_empty(),
                "{:?} should have at least one pathset",
                desc.name
            );
        }

        if paths.is_empty() || builder.config.include_default_paths {
            for (desc, should_run) in v.iter().zip(&should_runs) {
                if desc.default && should_run.is_really_default() {
                    for pathset in &should_run.paths {
                        desc.maybe_run(builder, pathset);
                    }
                }
            }
        }

        for path in paths {
            // strip CurDir prefix if present
            let path = match path.strip_prefix(".") {
                Ok(p) => p,
                Err(_) => path,
            };

            let mut attempted_run = false;
            for (desc, should_run) in v.iter().zip(&should_runs) {
                if let Some(suite) = should_run.is_suite_path(path) {
                    attempted_run = true;
                    desc.maybe_run(builder, suite);
                } else if let Some(pathset) = should_run.pathset_for_path(path) {
                    attempted_run = true;
                    desc.maybe_run(builder, pathset);
                }
            }

            if !attempted_run {
                panic!("error: no rules matched {}", path.display());
            }
        }
    }
}

enum ReallyDefault<'a> {
    Bool(bool),
    Lazy(Lazy<bool, Box<dyn Fn() -> bool + 'a>>),
}

pub struct ShouldRun<'a> {
    pub builder: &'a Builder<'a>,
    // use a BTreeSet to maintain sort order
    paths: BTreeSet<PathSet>,

    // If this is a default rule, this is an additional constraint placed on
    // its run. Generally something like compiler docs being enabled.
    is_really_default: ReallyDefault<'a>,
}

impl<'a> ShouldRun<'a> {
    fn new(builder: &'a Builder<'_>) -> ShouldRun<'a> {
        ShouldRun {
            builder,
            paths: BTreeSet::new(),
            is_really_default: ReallyDefault::Bool(true), // by default no additional conditions
        }
    }

    pub fn default_condition(mut self, cond: bool) -> Self {
        self.is_really_default = ReallyDefault::Bool(cond);
        self
    }

    pub fn lazy_default_condition(mut self, lazy_cond: Box<dyn Fn() -> bool + 'a>) -> Self {
        self.is_really_default = ReallyDefault::Lazy(Lazy::new(lazy_cond));
        self
    }

    pub fn is_really_default(&self) -> bool {
        match &self.is_really_default {
            ReallyDefault::Bool(val) => *val,
            ReallyDefault::Lazy(lazy) => *lazy.deref(),
        }
    }

    /// Indicates it should run if the command-line selects the given crate or
    /// any of its (local) dependencies.
    ///
    /// Compared to `krate`, this treats the dependencies as aliases for the
    /// same job. Generally it is preferred to use `krate`, and treat each
    /// individual path separately. For example `./x.py test src/liballoc`
    /// (which uses `krate`) will test just `liballoc`. However, `./x.py check
    /// src/liballoc` (which uses `all_krates`) will check all of `libtest`.
    /// `all_krates` should probably be removed at some point.
    pub fn all_krates(mut self, name: &str) -> Self {
        let mut set = BTreeSet::new();
        for krate in self.builder.in_tree_crates(name, None) {
            let path = krate.local_path(self.builder);
            set.insert(path);
        }
        self.paths.insert(PathSet::Set(set));
        self
    }

    /// Indicates it should run if the command-line selects the given crate or
    /// any of its (local) dependencies.
    ///
    /// `make_run` will be called separately for each matching command-line path.
    pub fn krate(mut self, name: &str) -> Self {
        for krate in self.builder.in_tree_crates(name, None) {
            let path = krate.local_path(self.builder);
            self.paths.insert(PathSet::one(path));
        }
        self
    }

    // single, non-aliased path
    pub fn path(self, path: &str) -> Self {
        self.paths(&[path])
    }

    // multiple aliases for the same job
    pub fn paths(mut self, paths: &[&str]) -> Self {
        self.paths.insert(PathSet::Set(paths.iter().map(PathBuf::from).collect()));
        self
    }

    pub fn is_suite_path(&self, path: &Path) -> Option<&PathSet> {
        self.paths.iter().find(|pathset| match pathset {
            PathSet::Suite(p) => path.starts_with(p),
            PathSet::Set(_) => false,
        })
    }

    pub fn suite_path(mut self, suite: &str) -> Self {
        self.paths.insert(PathSet::Suite(PathBuf::from(suite)));
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
    Clippy,
    Fix,
    Format,
    Test,
    Bench,
    Dist,
    Doc,
    Install,
    Run,
}

impl<'a> Builder<'a> {
    fn get_step_descriptions(kind: Kind) -> Vec<StepDescription> {
        macro_rules! describe {
            ($($rule:ty),+ $(,)?) => {{
                vec![$(StepDescription::from::<$rule>()),+]
            }};
        }
        match kind {
            Kind::Build => describe!(
                compile::Std,
                compile::Assemble,
                compile::CodegenBackend,
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
                tool::RustAnalyzer,
                tool::RustDemangler,
                tool::Rustdoc,
                tool::Clippy,
                tool::CargoClippy,
                native::Llvm,
                native::Sanitizers,
                tool::Rustfmt,
                tool::Miri,
                tool::CargoMiri,
                native::Lld,
                native::CrtBeginEnd
            ),
            Kind::Check | Kind::Clippy { .. } | Kind::Fix | Kind::Format => describe!(
                check::Std,
                check::Rustc,
                check::Rustdoc,
                check::CodegenBackend,
                check::Clippy,
                check::Miri,
                check::Rls,
                check::Rustfmt,
                check::Bootstrap
            ),
            Kind::Test => describe!(
                crate::toolstate::ToolStateCheck,
                test::ExpandYamlAnchors,
                test::Tidy,
                test::Ui,
                test::RunPassValgrind,
                test::MirOpt,
                test::Codegen,
                test::CodegenUnits,
                test::Assembly,
                test::Incremental,
                test::Debuginfo,
                test::UiFullDeps,
                test::Rustdoc,
                test::Pretty,
                test::Crate,
                test::CrateLibrustc,
                test::CrateRustdoc,
                test::CrateRustdocJsonTypes,
                test::Linkcheck,
                test::TierCheck,
                test::Cargotest,
                test::Cargo,
                test::Rls,
                test::ErrorIndex,
                test::Distcheck,
                test::RunMakeFullDeps,
                test::Nomicon,
                test::Reference,
                test::RustdocBook,
                test::RustByExample,
                test::TheBook,
                test::UnstableBook,
                test::RustcBook,
                test::LintDocs,
                test::RustcGuide,
                test::EmbeddedBook,
                test::EditionGuide,
                test::Rustfmt,
                test::Miri,
                test::Clippy,
                test::RustDemangler,
                test::CompiletestTest,
                test::RustdocJSStd,
                test::RustdocJSNotStd,
                test::RustdocGUI,
                test::RustdocTheme,
                test::RustdocUi,
                test::RustdocJson,
                test::HtmlCheck,
                // Run bootstrap close to the end as it's unlikely to fail
                test::Bootstrap,
                // Run run-make last, since these won't pass without make on Windows
                test::RunMake,
            ),
            Kind::Bench => describe!(test::Crate, test::CrateLibrustc),
            Kind::Doc => describe!(
                doc::UnstableBook,
                doc::UnstableBookGen,
                doc::TheBook,
                doc::Standalone,
                doc::Std,
                doc::Rustc,
                doc::Rustdoc,
                doc::Rustfmt,
                doc::ErrorIndex,
                doc::Nomicon,
                doc::Reference,
                doc::RustdocBook,
                doc::RustByExample,
                doc::RustcBook,
                doc::CargoBook,
                doc::EmbeddedBook,
                doc::EditionGuide,
            ),
            Kind::Dist => describe!(
                dist::Docs,
                dist::RustcDocs,
                dist::Mingw,
                dist::Rustc,
                dist::DebuggerScripts,
                dist::Std,
                dist::RustcDev,
                dist::Analysis,
                dist::Src,
                dist::PlainSourceTarball,
                dist::Cargo,
                dist::Rls,
                dist::RustAnalyzer,
                dist::Rustfmt,
                dist::RustDemangler,
                dist::Clippy,
                dist::Miri,
                dist::LlvmTools,
                dist::RustDev,
                dist::Extended,
                dist::BuildManifest,
                dist::ReproducibleArtifacts,
            ),
            Kind::Install => describe!(
                install::Docs,
                install::Std,
                install::Cargo,
                install::Rls,
                install::RustAnalyzer,
                install::Rustfmt,
                install::RustDemangler,
                install::Clippy,
                install::Miri,
                install::Analysis,
                install::Src,
                install::Rustc
            ),
            Kind::Run => describe!(run::ExpandYamlAnchors, run::BuildManifest, run::BumpStage0),
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

        let builder = Self::new_internal(build, kind, vec![]);
        let builder = &builder;
        let mut should_run = ShouldRun::new(builder);
        for desc in Builder::get_step_descriptions(builder.kind) {
            should_run = (desc.should_run)(should_run);
        }
        let mut help = String::from("Available paths:\n");
        let mut add_path = |path: &Path| {
            help.push_str(&format!("    ./x.py {} {}\n", subcommand, path.display()));
        };
        for pathset in should_run.paths {
            match pathset {
                PathSet::Set(set) => {
                    for path in set {
                        add_path(&path);
                    }
                }
                PathSet::Suite(path) => {
                    add_path(&path.join("..."));
                }
            }
        }
        Some(help)
    }

    fn new_internal(build: &Build, kind: Kind, paths: Vec<PathBuf>) -> Builder<'_> {
        Builder {
            build,
            top_stage: build.config.stage,
            kind,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
            time_spent_on_dependencies: Cell::new(Duration::new(0, 0)),
            paths,
        }
    }

    pub fn new(build: &Build) -> Builder<'_> {
        let (kind, paths) = match build.config.cmd {
            Subcommand::Build { ref paths } => (Kind::Build, &paths[..]),
            Subcommand::Check { ref paths } => (Kind::Check, &paths[..]),
            Subcommand::Clippy { ref paths, .. } => (Kind::Clippy, &paths[..]),
            Subcommand::Fix { ref paths } => (Kind::Fix, &paths[..]),
            Subcommand::Doc { ref paths, .. } => (Kind::Doc, &paths[..]),
            Subcommand::Test { ref paths, .. } => (Kind::Test, &paths[..]),
            Subcommand::Bench { ref paths, .. } => (Kind::Bench, &paths[..]),
            Subcommand::Dist { ref paths } => (Kind::Dist, &paths[..]),
            Subcommand::Install { ref paths } => (Kind::Install, &paths[..]),
            Subcommand::Run { ref paths } => (Kind::Run, &paths[..]),
            Subcommand::Format { .. } | Subcommand::Clean { .. } | Subcommand::Setup { .. } => {
                panic!()
            }
        };

        Self::new_internal(build, kind, paths.to_owned())
    }

    pub fn execute_cli(&self) {
        self.run_step_descriptions(&Builder::get_step_descriptions(self.kind), &self.paths);
    }

    pub fn default_doc(&self, paths: &[PathBuf]) {
        self.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), paths);
    }

    /// NOTE: keep this in sync with `rustdoc::clean::utils::doc_rust_lang_org_channel`, or tests will fail on beta/stable.
    pub fn doc_rust_lang_org_channel(&self) -> String {
        let channel = match &*self.config.channel {
            "stable" => &self.version,
            "beta" => "beta",
            "nightly" | "dev" => "nightly",
            // custom build of rustdoc maybe? link to the latest stable docs just in case
            _ => "stable",
        };
        "https://doc.rust-lang.org/".to_owned() + channel
    }

    fn run_step_descriptions(&self, v: &[StepDescription], paths: &[PathBuf]) {
        StepDescription::run(v, self, paths);
    }

    /// Obtain a compiler at a given stage and for a given host. Explicitly does
    /// not take `Compiler` since all `Compiler` instances are meant to be
    /// obtained through this function, since it ensures that they are valid
    /// (i.e., built and assembled).
    pub fn compiler(&self, stage: u32, host: TargetSelection) -> Compiler {
        self.ensure(compile::Assemble { target_compiler: Compiler { stage, host } })
    }

    /// Similar to `compiler`, except handles the full-bootstrap option to
    /// silently use the stage1 compiler instead of a stage2 compiler if one is
    /// requested.
    ///
    /// Note that this does *not* have the side effect of creating
    /// `compiler(stage, host)`, unlike `compiler` above which does have such
    /// a side effect. The returned compiler here can only be used to compile
    /// new artifacts, it can't be used to rely on the presence of a particular
    /// sysroot.
    ///
    /// See `force_use_stage1` for documentation on what each argument is.
    pub fn compiler_for(
        &self,
        stage: u32,
        host: TargetSelection,
        target: TargetSelection,
    ) -> Compiler {
        if self.build.force_use_stage1(Compiler { stage, host }, target) {
            self.compiler(1, self.config.build)
        } else {
            self.compiler(stage, host)
        }
    }

    pub fn sysroot(&self, compiler: Compiler) -> Interned<PathBuf> {
        self.ensure(compile::Sysroot { compiler })
    }

    /// Returns the libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
    pub fn sysroot_libdir(&self, compiler: Compiler, target: TargetSelection) -> Interned<PathBuf> {
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        struct Libdir {
            compiler: Compiler,
            target: TargetSelection,
        }
        impl Step for Libdir {
            type Output = Interned<PathBuf>;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.never()
            }

            fn run(self, builder: &Builder<'_>) -> Interned<PathBuf> {
                let lib = builder.sysroot_libdir_relative(self.compiler);
                let sysroot = builder
                    .sysroot(self.compiler)
                    .join(lib)
                    .join("rustlib")
                    .join(self.target.triple)
                    .join("lib");
                // Avoid deleting the rustlib/ directory we just copied
                // (in `impl Step for Sysroot`).
                if !builder.config.download_rustc {
                    let _ = fs::remove_dir_all(&sysroot);
                    t!(fs::create_dir_all(&sysroot));
                }
                INTERNER.intern_path(sysroot)
            }
        }
        self.ensure(Libdir { compiler, target })
    }

    pub fn sysroot_codegen_backends(&self, compiler: Compiler) -> PathBuf {
        self.sysroot_libdir(compiler, compiler.host).with_file_name("codegen-backends")
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
            match self.config.libdir_relative() {
                Some(relative_libdir) if compiler.stage >= 1 => {
                    self.sysroot(compiler).join(relative_libdir)
                }
                _ => self.sysroot(compiler).join(libdir(compiler.host)),
            }
        }
    }

    /// Returns the compiler's relative libdir where it stores the dynamic libraries that
    /// it itself links against.
    ///
    /// For example this returns `lib` on Unix and `bin` on
    /// Windows.
    pub fn libdir_relative(&self, compiler: Compiler) -> &Path {
        if compiler.is_snapshot(self) {
            libdir(self.config.build).as_ref()
        } else {
            match self.config.libdir_relative() {
                Some(relative_libdir) if compiler.stage >= 1 => relative_libdir,
                _ => libdir(compiler.host).as_ref(),
            }
        }
    }

    /// Returns the compiler's relative libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
    ///
    /// For example this returns `lib` on Unix and Windows.
    pub fn sysroot_libdir_relative(&self, compiler: Compiler) -> &Path {
        match self.config.libdir_relative() {
            Some(relative_libdir) if compiler.stage >= 1 => relative_libdir,
            _ if compiler.stage == 0 => &self.build.initial_libdir,
            _ => Path::new("lib"),
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

        let mut dylib_dirs = vec![self.rustc_libdir(compiler)];

        // Ensure that the downloaded LLVM libraries can be found.
        if self.config.llvm_from_ci {
            let ci_llvm_lib = self.out.join(&*compiler.host.triple).join("ci-llvm").join("lib");
            dylib_dirs.push(ci_llvm_lib);
        }

        add_dylib_path(dylib_dirs, cmd);
    }

    /// Gets a path to the compiler specified.
    pub fn rustc(&self, compiler: Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.initial_rustc.clone()
        } else {
            self.sysroot(compiler).join("bin").join(exe("rustc", compiler.host))
        }
    }

    /// Gets the paths to all of the compiler's codegen backends.
    fn codegen_backends(&self, compiler: Compiler) -> impl Iterator<Item = PathBuf> {
        fs::read_dir(self.sysroot_codegen_backends(compiler))
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
    }

    pub fn rustdoc(&self, compiler: Compiler) -> PathBuf {
        self.ensure(tool::Rustdoc { compiler })
    }

    pub fn rustdoc_cmd(&self, compiler: Compiler) -> Command {
        let mut cmd = Command::new(&self.out.join("bootstrap/debug/rustdoc"));
        cmd.env("RUSTC_STAGE", compiler.stage.to_string())
            .env("RUSTC_SYSROOT", self.sysroot(compiler))
            // Note that this is *not* the sysroot_libdir because rustdoc must be linked
            // equivalently to rustc.
            .env("RUSTDOC_LIBDIR", self.rustc_libdir(compiler))
            .env("CFG_RELEASE_CHANNEL", &self.config.channel)
            .env("RUSTDOC_REAL", self.rustdoc(compiler))
            .env("RUSTC_BOOTSTRAP", "1");

        cmd.arg("-Wrustdoc::invalid_codeblock_attributes");

        if self.config.deny_warnings {
            cmd.arg("-Dwarnings");
        }
        cmd.arg("-Znormalize-docs");

        // Remove make-related flags that can cause jobserver problems.
        cmd.env_remove("MAKEFLAGS");
        cmd.env_remove("MFLAGS");

        if let Some(linker) = self.linker(compiler.host) {
            cmd.env("RUSTDOC_LINKER", linker);
        }
        if self.is_fuse_ld_lld(compiler.host) {
            cmd.env("RUSTDOC_FUSE_LD_LLD", "1");
        }
        cmd
    }

    /// Return the path to `llvm-config` for the target, if it exists.
    ///
    /// Note that this returns `None` if LLVM is disabled, or if we're in a
    /// check build or dry-run, where there's no need to build all of LLVM.
    fn llvm_config(&self, target: TargetSelection) -> Option<PathBuf> {
        if self.config.llvm_enabled() && self.kind != Kind::Check && !self.config.dry_run {
            let llvm_config = self.ensure(native::Llvm { target });
            if llvm_config.is_file() {
                return Some(llvm_config);
            }
        }
        None
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
        source_type: SourceType,
        target: TargetSelection,
        cmd: &str,
    ) -> Cargo {
        let mut cargo = Command::new(&self.initial_cargo);
        let out_dir = self.stage_out(compiler, mode);

        // Codegen backends are not yet tracked by -Zbinary-dep-depinfo,
        // so we need to explicitly clear out if they've been updated.
        for backend in self.codegen_backends(compiler) {
            self.clear_if_dirty(&out_dir, &backend);
        }

        if cmd == "doc" || cmd == "rustdoc" {
            let my_out = match mode {
                // This is the intended out directory for compiler documentation.
                Mode::Rustc | Mode::ToolRustc => self.compiler_doc_out(target),
                Mode::Std => out_dir.join(target.triple).join("doc"),
                _ => panic!("doc mode {:?} not expected", mode),
            };
            let rustdoc = self.rustdoc(compiler);
            self.clear_if_dirty(&my_out, &rustdoc);
        }

        cargo.env("CARGO_TARGET_DIR", &out_dir).arg(cmd);

        let profile_var = |name: &str| {
            let profile = if self.config.rust_optimize { "RELEASE" } else { "DEV" };
            format!("CARGO_PROFILE_{}_{}", profile, name)
        };

        // See comment in rustc_llvm/build.rs for why this is necessary, largely llvm-config
        // needs to not accidentally link to libLLVM in stage0/lib.
        cargo.env("REAL_LIBRARY_PATH_VAR", &util::dylib_path_var());
        if let Some(e) = env::var_os(util::dylib_path_var()) {
            cargo.env("REAL_LIBRARY_PATH", e);
        }

        // Found with `rg "init_env_logger\("`. If anyone uses `init_env_logger`
        // from out of tree it shouldn't matter, since x.py is only used for
        // building in-tree.
        let color_logs = ["RUSTDOC_LOG_COLOR", "RUSTC_LOG_COLOR", "RUST_LOG_COLOR"];
        match self.build.config.color {
            Color::Always => {
                cargo.arg("--color=always");
                for log in &color_logs {
                    cargo.env(log, "always");
                }
            }
            Color::Never => {
                cargo.arg("--color=never");
                for log in &color_logs {
                    cargo.env(log, "never");
                }
            }
            Color::Auto => {} // nothing to do
        }

        if cmd != "install" {
            cargo.arg("--target").arg(target.rustc_target_arg());
        } else {
            assert_eq!(target, compiler.host);
        }

        // Set a flag for `check`/`clippy`/`fix`, so that certain build
        // scripts can do less work (i.e. not building/requiring LLVM).
        if cmd == "check" || cmd == "clippy" || cmd == "fix" {
            // If we've not yet built LLVM, or it's stale, then bust
            // the rustc_llvm cache. That will always work, even though it
            // may mean that on the next non-check build we'll need to rebuild
            // rustc_llvm. But if LLVM is stale, that'll be a tiny amount
            // of work comparitively, and we'd likely need to rebuild it anyway,
            // so that's okay.
            if crate::native::prebuilt_llvm_config(self, target).is_err() {
                cargo.env("RUST_CHECK", "1");
            }
        }

        let stage = if compiler.stage == 0 && self.local_rebuild {
            // Assume the local-rebuild rustc already has stage1 features.
            1
        } else {
            compiler.stage
        };

        let mut rustflags = Rustflags::new(target);
        if stage != 0 {
            if let Ok(s) = env::var("CARGOFLAGS_NOT_BOOTSTRAP") {
                cargo.args(s.split_whitespace());
            }
            rustflags.env("RUSTFLAGS_NOT_BOOTSTRAP");
        } else {
            if let Ok(s) = env::var("CARGOFLAGS_BOOTSTRAP") {
                cargo.args(s.split_whitespace());
            }
            rustflags.env("RUSTFLAGS_BOOTSTRAP");
            if cmd == "clippy" {
                // clippy overwrites sysroot if we pass it to cargo.
                // Pass it directly to clippy instead.
                // NOTE: this can't be fixed in clippy because we explicitly don't set `RUSTC`,
                // so it has no way of knowing the sysroot.
                rustflags.arg("--sysroot");
                rustflags.arg(
                    self.sysroot(compiler)
                        .as_os_str()
                        .to_str()
                        .expect("sysroot must be valid UTF-8"),
                );
                // Only run clippy on a very limited subset of crates (in particular, not build scripts).
                cargo.arg("-Zunstable-options");
                // Explicitly does *not* set `--cfg=bootstrap`, since we're using a nightly clippy.
                let host_version = Command::new("rustc").arg("--version").output().map_err(|_| ());
                let output = host_version.and_then(|output| {
                    if output.status.success() {
                        Ok(output)
                    } else {
                        Err(())
                    }
                }).unwrap_or_else(|_| {
                    eprintln!(
                        "error: `x.py clippy` requires a host `rustc` toolchain with the `clippy` component"
                    );
                    eprintln!("help: try `rustup component add clippy`");
                    std::process::exit(1);
                });
                if !t!(std::str::from_utf8(&output.stdout)).contains("nightly") {
                    rustflags.arg("--cfg=bootstrap");
                }
            } else {
                rustflags.arg("--cfg=bootstrap");
            }
        }

        let use_new_symbol_mangling = match self.config.rust_new_symbol_mangling {
            Some(setting) => {
                // If an explicit setting is given, use that
                setting
            }
            None => {
                if mode == Mode::Std {
                    // The standard library defaults to the legacy scheme
                    false
                } else {
                    // The compiler and tools default to the new scheme
                    true
                }
            }
        };

        if use_new_symbol_mangling {
            rustflags.arg("-Zsymbol-mangling-version=v0");
        } else {
            rustflags.arg("-Zsymbol-mangling-version=legacy");
        }

        // FIXME: It might be better to use the same value for both `RUSTFLAGS` and `RUSTDOCFLAGS`,
        // but this breaks CI. At the very least, stage0 `rustdoc` needs `--cfg bootstrap`. See
        // #71458.
        let mut rustdocflags = rustflags.clone();
        rustdocflags.propagate_cargo_env("RUSTDOCFLAGS");
        if stage == 0 {
            rustdocflags.env("RUSTDOCFLAGS_BOOTSTRAP");
        } else {
            rustdocflags.env("RUSTDOCFLAGS_NOT_BOOTSTRAP");
        }

        if let Ok(s) = env::var("CARGOFLAGS") {
            cargo.args(s.split_whitespace());
        }

        match mode {
            Mode::Std | Mode::ToolBootstrap | Mode::ToolStd => {}
            Mode::Rustc | Mode::Codegen | Mode::ToolRustc => {
                // Build proc macros both for the host and the target
                if target != compiler.host && cmd != "check" {
                    cargo.arg("-Zdual-proc-macros");
                    rustflags.arg("-Zdual-proc-macros");
                }
            }
        }

        // This tells Cargo (and in turn, rustc) to output more complete
        // dependency information.  Most importantly for rustbuild, this
        // includes sysroot artifacts, like libstd, which means that we don't
        // need to track those in rustbuild (an error prone process!). This
        // feature is currently unstable as there may be some bugs and such, but
        // it represents a big improvement in rustbuild's reliability on
        // rebuilds, so we're using it here.
        //
        // For some additional context, see #63470 (the PR originally adding
        // this), as well as #63012 which is the tracking issue for this
        // feature on the rustc side.
        cargo.arg("-Zbinary-dep-depinfo");

        cargo.arg("-j").arg(self.jobs().to_string());
        // Remove make-related flags to ensure Cargo can correctly set things up
        cargo.env_remove("MAKEFLAGS");
        cargo.env_remove("MFLAGS");

        // FIXME: Temporary fix for https://github.com/rust-lang/cargo/issues/3005
        // Force cargo to output binaries with disambiguating hashes in the name
        let mut metadata = if compiler.stage == 0 {
            // Treat stage0 like a special channel, whether it's a normal prior-
            // release rustc or a local rebuild with the same version, so we
            // never mix these libraries by accident.
            "bootstrap".to_string()
        } else {
            self.config.channel.to_string()
        };
        // We want to make sure that none of the dependencies between
        // std/test/rustc unify with one another. This is done for weird linkage
        // reasons but the gist of the problem is that if librustc, libtest, and
        // libstd all depend on libc from crates.io (which they actually do) we
        // want to make sure they all get distinct versions. Things get really
        // weird if we try to unify all these dependencies right now, namely
        // around how many times the library is linked in dynamic libraries and
        // such. If rustc were a static executable or if we didn't ship dylibs
        // this wouldn't be a problem, but we do, so it is. This is in general
        // just here to make sure things build right. If you can remove this and
        // things still build right, please do!
        match mode {
            Mode::Std => metadata.push_str("std"),
            // When we're building rustc tools, they're built with a search path
            // that contains things built during the rustc build. For example,
            // bitflags is built during the rustc build, and is a dependency of
            // rustdoc as well. We're building rustdoc in a different target
            // directory, though, which means that Cargo will rebuild the
            // dependency. When we go on to build rustdoc, we'll look for
            // bitflags, and find two different copies: one built during the
            // rustc step and one that we just built. This isn't always a
            // problem, somehow -- not really clear why -- but we know that this
            // fixes things.
            Mode::ToolRustc => metadata.push_str("tool-rustc"),
            // Same for codegen backends.
            Mode::Codegen => metadata.push_str("codegen"),
            _ => {}
        }
        cargo.env("__CARGO_DEFAULT_LIB_METADATA", &metadata);

        if cmd == "clippy" {
            rustflags.arg("-Zforce-unstable-if-unmarked");
        }

        rustflags.arg("-Zmacro-backtrace");

        let want_rustdoc = self.doc_tests != DocTests::No;

        // We synthetically interpret a stage0 compiler used to build tools as a
        // "raw" compiler in that it's the exact snapshot we download. Normally
        // the stage0 build means it uses libraries build by the stage0
        // compiler, but for tools we just use the precompiled libraries that
        // we've downloaded
        let use_snapshot = mode == Mode::ToolBootstrap;
        assert!(!use_snapshot || stage == 0 || self.local_rebuild);

        let maybe_sysroot = self.sysroot(compiler);
        let sysroot = if use_snapshot { self.rustc_snapshot_sysroot() } else { &maybe_sysroot };
        let libdir = self.rustc_libdir(compiler);

        // Clear the output directory if the real rustc we're using has changed;
        // Cargo cannot detect this as it thinks rustc is bootstrap/debug/rustc.
        //
        // Avoid doing this during dry run as that usually means the relevant
        // compiler is not yet linked/copied properly.
        //
        // Only clear out the directory if we're compiling std; otherwise, we
        // should let Cargo take care of things for us (via depdep info)
        if !self.config.dry_run && mode == Mode::Std && cmd == "build" {
            self.clear_if_dirty(&out_dir, &self.rustc(compiler));
        }

        // Customize the compiler we're running. Specify the compiler to cargo
        // as our shim and then pass it some various options used to configure
        // how the actual compiler itself is called.
        //
        // These variables are primarily all read by
        // src/bootstrap/bin/{rustc.rs,rustdoc.rs}
        cargo
            .env("RUSTBUILD_NATIVE_DIR", self.native_dir(target))
            .env("RUSTC_REAL", self.rustc(compiler))
            .env("RUSTC_STAGE", stage.to_string())
            .env("RUSTC_SYSROOT", &sysroot)
            .env("RUSTC_LIBDIR", &libdir)
            .env("RUSTDOC", self.out.join("bootstrap/debug/rustdoc"))
            .env(
                "RUSTDOC_REAL",
                if cmd == "doc" || cmd == "rustdoc" || (cmd == "test" && want_rustdoc) {
                    self.rustdoc(compiler)
                } else {
                    PathBuf::from("/path/to/nowhere/rustdoc/not/required")
                },
            )
            .env("RUSTC_ERROR_METADATA_DST", self.extended_error_dir())
            .env("RUSTC_BREAK_ON_ICE", "1");
        // Clippy support is a hack and uses the default `cargo-clippy` in path.
        // Don't override RUSTC so that the `cargo-clippy` in path will be run.
        if cmd != "clippy" {
            cargo.env("RUSTC", self.out.join("bootstrap/debug/rustc"));
        }

        // Dealing with rpath here is a little special, so let's go into some
        // detail. First off, `-rpath` is a linker option on Unix platforms
        // which adds to the runtime dynamic loader path when looking for
        // dynamic libraries. We use this by default on Unix platforms to ensure
        // that our nightlies behave the same on Windows, that is they work out
        // of the box. This can be disabled, of course, but basically that's why
        // we're gated on RUSTC_RPATH here.
        //
        // Ok, so the astute might be wondering "why isn't `-C rpath` used
        // here?" and that is indeed a good question to ask. This codegen
        // option is the compiler's current interface to generating an rpath.
        // Unfortunately it doesn't quite suffice for us. The flag currently
        // takes no value as an argument, so the compiler calculates what it
        // should pass to the linker as `-rpath`. This unfortunately is based on
        // the **compile time** directory structure which when building with
        // Cargo will be very different than the runtime directory structure.
        //
        // All that's a really long winded way of saying that if we use
        // `-Crpath` then the executables generated have the wrong rpath of
        // something like `$ORIGIN/deps` when in fact the way we distribute
        // rustc requires the rpath to be `$ORIGIN/../lib`.
        //
        // So, all in all, to set up the correct rpath we pass the linker
        // argument manually via `-C link-args=-Wl,-rpath,...`. Plus isn't it
        // fun to pass a flag to a tool to pass a flag to pass a flag to a tool
        // to change a flag in a binary?
        if self.config.rust_rpath && util::use_host_linker(target) {
            let rpath = if target.contains("apple") {
                // Note that we need to take one extra step on macOS to also pass
                // `-Wl,-instal_name,@rpath/...` to get things to work right. To
                // do that we pass a weird flag to the compiler to get it to do
                // so. Note that this is definitely a hack, and we should likely
                // flesh out rpath support more fully in the future.
                rustflags.arg("-Zosx-rpath-install-name");
                Some("-Wl,-rpath,@loader_path/../lib")
            } else if !target.contains("windows") {
                Some("-Wl,-rpath,$ORIGIN/../lib")
            } else {
                None
            };
            if let Some(rpath) = rpath {
                rustflags.arg(&format!("-Clink-args={}", rpath));
            }
        }

        if let Some(host_linker) = self.linker(compiler.host) {
            cargo.env("RUSTC_HOST_LINKER", host_linker);
        }
        if self.is_fuse_ld_lld(compiler.host) {
            cargo.env("RUSTC_HOST_FUSE_LD_LLD", "1");
            cargo.env("RUSTDOC_FUSE_LD_LLD", "1");
        }

        if let Some(target_linker) = self.linker(target) {
            let target = crate::envify(&target.triple);
            cargo.env(&format!("CARGO_TARGET_{}_LINKER", target), target_linker);
        }
        if self.is_fuse_ld_lld(target) {
            rustflags.arg("-Clink-args=-fuse-ld=lld");
        }
        self.lld_flags(target).for_each(|flag| {
            rustdocflags.arg(&flag);
        });

        if !(["build", "check", "clippy", "fix", "rustc"].contains(&cmd)) && want_rustdoc {
            cargo.env("RUSTDOC_LIBDIR", self.rustc_libdir(compiler));
        }

        let debuginfo_level = match mode {
            Mode::Rustc | Mode::Codegen => self.config.rust_debuginfo_level_rustc,
            Mode::Std => self.config.rust_debuginfo_level_std,
            Mode::ToolBootstrap | Mode::ToolStd | Mode::ToolRustc => {
                self.config.rust_debuginfo_level_tools
            }
        };
        cargo.env(profile_var("DEBUG"), debuginfo_level.to_string());
        cargo.env(
            profile_var("DEBUG_ASSERTIONS"),
            if mode == Mode::Std {
                self.config.rust_debug_assertions_std.to_string()
            } else {
                self.config.rust_debug_assertions.to_string()
            },
        );
        cargo.env(
            profile_var("OVERFLOW_CHECKS"),
            if mode == Mode::Std {
                self.config.rust_overflow_checks_std.to_string()
            } else {
                self.config.rust_overflow_checks.to_string()
            },
        );

        // `dsymutil` adds time to builds on Apple platforms for no clear benefit, and also makes
        // it more difficult for debuggers to find debug info. The compiler currently defaults to
        // running `dsymutil` to preserve its historical default, but when compiling the compiler
        // itself, we skip it by default since we know it's safe to do so in that case.
        // See https://github.com/rust-lang/rust/issues/79361 for more info on this flag.
        if target.contains("apple") {
            if self.config.rust_run_dsymutil {
                rustflags.arg("-Csplit-debuginfo=packed");
            } else {
                rustflags.arg("-Csplit-debuginfo=unpacked");
            }
        }

        if self.config.cmd.bless() {
            // Bless `expect!` tests.
            cargo.env("UPDATE_EXPECT", "1");
        }

        if !mode.is_tool() {
            cargo.env("RUSTC_FORCE_UNSTABLE", "1");
        }

        if let Some(x) = self.crt_static(target) {
            if x {
                rustflags.arg("-Ctarget-feature=+crt-static");
            } else {
                rustflags.arg("-Ctarget-feature=-crt-static");
            }
        }

        if let Some(x) = self.crt_static(compiler.host) {
            cargo.env("RUSTC_HOST_CRT_STATIC", x.to_string());
        }

        if let Some(map_to) = self.build.debuginfo_map_to(GitRepo::Rustc) {
            let map = format!("{}={}", self.build.src.display(), map_to);
            cargo.env("RUSTC_DEBUGINFO_MAP", map);

            // `rustc` needs to know the virtual `/rustc/$hash` we're mapping to,
            // in order to opportunistically reverse it later.
            cargo.env("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR", map_to);
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
        if mode == Mode::Std {
            cargo
                .env("RUSTC_SNAPSHOT", &self.initial_rustc)
                .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_snapshot_libdir());
        } else {
            cargo
                .env("RUSTC_SNAPSHOT", self.rustc(compiler))
                .env("RUSTC_SNAPSHOT_LIBDIR", self.rustc_libdir(compiler));
        }

        // Tools that use compiler libraries may inherit the `-lLLVM` link
        // requirement, but the `-L` library path is not propagated across
        // separate Cargo projects. We can add LLVM's library path to the
        // platform-specific environment variable as a workaround.
        if mode == Mode::ToolRustc || mode == Mode::Codegen {
            if let Some(llvm_config) = self.llvm_config(target) {
                let llvm_libdir = output(Command::new(&llvm_config).arg("--libdir"));
                add_link_lib_path(vec![llvm_libdir.trim().into()], &mut cargo);
            }
        }

        // Compile everything except libraries and proc macros with the more
        // efficient initial-exec TLS model. This doesn't work with `dlopen`,
        // so we can't use it by default in general, but we can use it for tools
        // and our own internal libraries.
        if !mode.must_support_dlopen() && !target.triple.starts_with("powerpc-") {
            rustflags.arg("-Ztls-model=initial-exec");
        }

        if self.config.incremental {
            cargo.env("CARGO_INCREMENTAL", "1");
        } else {
            // Don't rely on any default setting for incr. comp. in Cargo
            cargo.env("CARGO_INCREMENTAL", "0");
        }

        if let Some(ref on_fail) = self.config.on_fail {
            cargo.env("RUSTC_ON_FAIL", on_fail);
        }

        if self.config.print_step_timings {
            cargo.env("RUSTC_PRINT_STEP_TIMINGS", "1");
        }

        if self.config.print_step_rusage {
            cargo.env("RUSTC_PRINT_STEP_RUSAGE", "1");
        }

        if self.config.backtrace_on_ice {
            cargo.env("RUSTC_BACKTRACE_ON_ICE", "1");
        }

        cargo.env("RUSTC_VERBOSE", self.verbosity.to_string());

        if source_type == SourceType::InTree {
            let mut lint_flags = Vec::new();
            // When extending this list, add the new lints to the RUSTFLAGS of the
            // build_bootstrap function of src/bootstrap/bootstrap.py as well as
            // some code doesn't go through this `rustc` wrapper.
            lint_flags.push("-Wrust_2018_idioms");
            lint_flags.push("-Wunused_lifetimes");
            lint_flags.push("-Wsemicolon_in_expressions_from_macros");

            if self.config.deny_warnings {
                lint_flags.push("-Dwarnings");
                rustdocflags.arg("-Dwarnings");
            }

            // This does not use RUSTFLAGS due to caching issues with Cargo.
            // Clippy is treated as an "in tree" tool, but shares the same
            // cache as other "submodule" tools. With these options set in
            // RUSTFLAGS, that causes *every* shared dependency to be rebuilt.
            // By injecting this into the rustc wrapper, this circumvents
            // Cargo's fingerprint detection. This is fine because lint flags
            // are always ignored in dependencies. Eventually this should be
            // fixed via better support from Cargo.
            cargo.env("RUSTC_LINT_FLAGS", lint_flags.join(" "));

            rustdocflags.arg("-Wrustdoc::invalid_codeblock_attributes");
        }

        if mode == Mode::Rustc {
            rustflags.arg("-Zunstable-options");
            rustflags.arg("-Wrustc::internal");
        }

        // Throughout the build Cargo can execute a number of build scripts
        // compiling C/C++ code and we need to pass compilers, archivers, flags, etc
        // obtained previously to those build scripts.
        // Build scripts use either the `cc` crate or `configure/make` so we pass
        // the options through environment variables that are fetched and understood by both.
        //
        // FIXME: the guard against msvc shouldn't need to be here
        if target.contains("msvc") {
            if let Some(ref cl) = self.config.llvm_clang_cl {
                cargo.env("CC", cl).env("CXX", cl);
            }
        } else {
            let ccache = self.config.ccache.as_ref();
            let ccacheify = |s: &Path| {
                let ccache = match ccache {
                    Some(ref s) => s,
                    None => return s.display().to_string(),
                };
                // FIXME: the cc-rs crate only recognizes the literal strings
                // `ccache` and `sccache` when doing caching compilations, so we
                // mirror that here. It should probably be fixed upstream to
                // accept a new env var or otherwise work with custom ccache
                // vars.
                match &ccache[..] {
                    "ccache" | "sccache" => format!("{} {}", ccache, s.display()),
                    _ => s.display().to_string(),
                }
            };
            let cc = ccacheify(&self.cc(target));
            cargo.env(format!("CC_{}", target.triple), &cc);

            let cflags = self.cflags(target, GitRepo::Rustc).join(" ");
            cargo.env(format!("CFLAGS_{}", target.triple), &cflags);

            if let Some(ar) = self.ar(target) {
                let ranlib = format!("{} s", ar.display());
                cargo
                    .env(format!("AR_{}", target.triple), ar)
                    .env(format!("RANLIB_{}", target.triple), ranlib);
            }

            if let Ok(cxx) = self.cxx(target) {
                let cxx = ccacheify(&cxx);
                cargo
                    .env(format!("CXX_{}", target.triple), &cxx)
                    .env(format!("CXXFLAGS_{}", target.triple), cflags);
            }
        }

        if mode == Mode::Std && self.config.extended && compiler.is_final_stage(self) {
            rustflags.arg("-Zsave-analysis");
            cargo.env(
                "RUST_SAVE_ANALYSIS_CONFIG",
                "{\"output_file\": null,\"full_docs\": false,\
                       \"pub_only\": true,\"reachable_only\": false,\
                       \"distro_crate\": true,\"signatures\": false,\"borrow_data\": false}",
            );
        }

        // If Control Flow Guard is enabled, pass the `control-flow-guard` flag to rustc
        // when compiling the standard library, since this might be linked into the final outputs
        // produced by rustc. Since this mitigation is only available on Windows, only enable it
        // for the standard library in case the compiler is run on a non-Windows platform.
        // This is not needed for stage 0 artifacts because these will only be used for building
        // the stage 1 compiler.
        if cfg!(windows)
            && mode == Mode::Std
            && self.config.control_flow_guard
            && compiler.stage >= 1
        {
            rustflags.arg("-Ccontrol-flow-guard");
        }

        // For `cargo doc` invocations, make rustdoc print the Rust version into the docs
        // This replaces spaces with newlines because RUSTDOCFLAGS does not
        // support arguments with regular spaces. Hopefully someday Cargo will
        // have space support.
        let rust_version = self.rust_version().replace(' ', "\n");
        rustdocflags.arg("--crate-version").arg(&rust_version);

        // Environment variables *required* throughout the build
        //
        // FIXME: should update code to not require this env var
        cargo.env("CFG_COMPILER_HOST_TRIPLE", target.triple);

        // Set this for all builds to make sure doc builds also get it.
        cargo.env("CFG_RELEASE_CHANNEL", &self.config.channel);

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
        // the binaries we ship of things like rustc_codegen_llvm (aka the rustc_codegen_llvm
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
        if !mode.is_tool() {
            cargo.env("WINAPI_NO_BUNDLED_LIBRARIES", "1");
        }

        for _ in 0..self.verbosity {
            cargo.arg("-v");
        }

        match (mode, self.config.rust_codegen_units_std, self.config.rust_codegen_units) {
            (Mode::Std, Some(n), _) | (_, _, Some(n)) => {
                cargo.env(profile_var("CODEGEN_UNITS"), n.to_string());
            }
            _ => {
                // Don't set anything
            }
        }

        if self.config.rust_optimize {
            // FIXME: cargo bench/install do not accept `--release`
            if cmd != "bench" && cmd != "install" {
                cargo.arg("--release");
            }
        }

        if self.config.locked_deps {
            cargo.arg("--locked");
        }
        if self.config.vendor || self.is_sudo {
            cargo.arg("--frozen");
        }

        // Try to use a sysroot-relative bindir, in case it was configured absolutely.
        cargo.env("RUSTC_INSTALL_BINDIR", self.config.bindir_relative());

        self.ci_env.force_coloring_in_ci(&mut cargo);

        // When we build Rust dylibs they're all intended for intermediate
        // usage, so make sure we pass the -Cprefer-dynamic flag instead of
        // linking all deps statically into the dylib.
        if matches!(mode, Mode::Std | Mode::Rustc) {
            rustflags.arg("-Cprefer-dynamic");
        }

        // When building incrementally we default to a lower ThinLTO import limit
        // (unless explicitly specified otherwise). This will produce a somewhat
        // slower code but give way better compile times.
        {
            let limit = match self.config.rust_thin_lto_import_instr_limit {
                Some(limit) => Some(limit),
                None if self.config.incremental => Some(10),
                _ => None,
            };

            if let Some(limit) = limit {
                rustflags.arg(&format!("-Cllvm-args=-import-instr-limit={}", limit));
            }
        }

        Cargo { command: cargo, rustflags, rustdocflags }
    }

    /// Ensure that a given step is built, returning its output. This will
    /// cache the step, so it is safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are built.
    pub fn ensure<S: Step>(&'a self, step: S) -> S::Output {
        {
            let mut stack = self.stack.borrow_mut();
            for stack_step in stack.iter() {
                // should skip
                if stack_step.downcast_ref::<S>().map_or(true, |stack_step| *stack_step != step) {
                    continue;
                }
                let mut out = String::new();
                out += &format!("\n\nCycle in build detected when adding {:?}\n", step);
                for el in stack.iter().rev() {
                    out += &format!("\t{:?}\n", el);
                }
                panic!("{}", out);
            }
            if let Some(out) = self.cache.get(&step) {
                self.verbose(&format!("{}c {:?}", "  ".repeat(stack.len()), step));

                return out;
            }
            self.verbose(&format!("{}> {:?}", "  ".repeat(stack.len()), step));
            stack.push(Box::new(step.clone()));
        }

        let (out, dur) = {
            let start = Instant::now();
            let zero = Duration::new(0, 0);
            let parent = self.time_spent_on_dependencies.replace(zero);
            let out = step.clone().run(self);
            let dur = start.elapsed();
            let deps = self.time_spent_on_dependencies.replace(parent + dur);
            (out, dur - deps)
        };

        if self.config.print_step_timings && !self.config.dry_run {
            println!("[TIMING] {:?} -- {}.{:03}", step, dur.as_secs(), dur.subsec_millis());
        }

        {
            let mut stack = self.stack.borrow_mut();
            let cur_step = stack.pop().expect("step stack empty");
            assert_eq!(cur_step.downcast_ref(), Some(&step));
        }
        self.verbose(&format!("{}< {:?}", "  ".repeat(self.stack.borrow().len()), step));
        self.cache.put(step, out.clone());
        out
    }

    /// Ensure that a given step is built *only if it's supposed to be built by default*, returning
    /// its output. This will cache the step, so it's safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are build.
    pub(crate) fn ensure_if_default<T, S: Step<Output = Option<T>>>(
        &'a self,
        step: S,
    ) -> S::Output {
        let desc = StepDescription::from::<S>();
        let should_run = (desc.should_run)(ShouldRun::new(self));

        // Avoid running steps contained in --exclude
        for pathset in &should_run.paths {
            if desc.is_excluded(self, pathset) {
                return None;
            }
        }

        // Only execute if it's supposed to run as default
        if desc.default && should_run.is_really_default() { self.ensure(step) } else { None }
    }

    /// Checks if any of the "should_run" paths is in the `Builder` paths.
    pub(crate) fn was_invoked_explicitly<S: Step>(&'a self) -> bool {
        let desc = StepDescription::from::<S>();
        let should_run = (desc.should_run)(ShouldRun::new(self));

        for path in &self.paths {
            if should_run.paths.iter().any(|s| s.has(path))
                && !desc.is_excluded(self, &PathSet::Suite(path.clone()))
            {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
struct Rustflags(String, TargetSelection);

impl Rustflags {
    fn new(target: TargetSelection) -> Rustflags {
        let mut ret = Rustflags(String::new(), target);
        ret.propagate_cargo_env("RUSTFLAGS");
        ret
    }

    /// By default, cargo will pick up on various variables in the environment. However, bootstrap
    /// reuses those variables to pass additional flags to rustdoc, so by default they get overriden.
    /// Explicitly add back any previous value in the environment.
    ///
    /// `prefix` is usually `RUSTFLAGS` or `RUSTDOCFLAGS`.
    fn propagate_cargo_env(&mut self, prefix: &str) {
        // Inherit `RUSTFLAGS` by default ...
        self.env(prefix);

        // ... and also handle target-specific env RUSTFLAGS if they're configured.
        let target_specific = format!("CARGO_TARGET_{}_{}", crate::envify(&self.1.triple), prefix);
        self.env(&target_specific);
    }

    fn env(&mut self, env: &str) {
        if let Ok(s) = env::var(env) {
            for part in s.split(' ') {
                self.arg(part);
            }
        }
    }

    fn arg(&mut self, arg: &str) -> &mut Self {
        assert_eq!(arg.split(' ').count(), 1);
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        self.0.push_str(arg);
        self
    }
}

#[derive(Debug)]
pub struct Cargo {
    command: Command,
    rustflags: Rustflags,
    rustdocflags: Rustflags,
}

impl Cargo {
    pub fn rustdocflag(&mut self, arg: &str) -> &mut Cargo {
        self.rustdocflags.arg(arg);
        self
    }
    pub fn rustflag(&mut self, arg: &str) -> &mut Cargo {
        self.rustflags.arg(arg);
        self
    }

    pub fn arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Cargo {
        self.command.arg(arg.as_ref());
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Cargo
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.arg(arg.as_ref());
        }
        self
    }

    pub fn env(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> &mut Cargo {
        // These are managed through rustflag/rustdocflag interfaces.
        assert_ne!(key.as_ref(), "RUSTFLAGS");
        assert_ne!(key.as_ref(), "RUSTDOCFLAGS");
        self.command.env(key.as_ref(), value.as_ref());
        self
    }

    pub fn add_rustc_lib_path(&mut self, builder: &Builder<'_>, compiler: Compiler) {
        builder.add_rustc_lib_path(compiler, &mut self.command);
    }

    pub fn current_dir(&mut self, dir: &Path) -> &mut Cargo {
        self.command.current_dir(dir);
        self
    }
}

impl From<Cargo> for Command {
    fn from(mut cargo: Cargo) -> Command {
        let rustflags = &cargo.rustflags.0;
        if !rustflags.is_empty() {
            cargo.command.env("RUSTFLAGS", rustflags);
        }

        let rustdocflags = &cargo.rustdocflags.0;
        if !rustdocflags.is_empty() {
            cargo.command.env("RUSTDOCFLAGS", rustdocflags);
        }

        cargo.command
    }
}
