use std::any::{Any, type_name};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::fmt::{Debug, Write};
use std::hash::Hash;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::{env, fs, iter};

use clap::ValueEnum;
#[cfg(feature = "tracing")]
use tracing::instrument;

pub(crate) use self::cargo::{Cargo, apply_pgo, cargo_profile_var};
use crate::core::build_steps::compile::{Std, StdLink, looks_like_codegen_backend};
use crate::core::build_steps::tool::RustcPrivateCompilers;
use crate::core::build_steps::{
    check, clean, clippy, compile, dist, doc, gcc, install, llvm, run, setup, test, tool, vendor,
};
use crate::core::builder::step_stack::StepRecord;
pub use crate::core::builder::step_stack::StepStack;
use crate::core::compiler::Compiler;
use crate::core::config::flags::Subcommand;
use crate::core::config::{DryRun, TargetSelection};
use crate::core::metadata::Crate;
use crate::core::session::Session;
use crate::trace;
use crate::utils::build_stamp::BuildStamp;
use crate::utils::cache::Cache;
use crate::utils::exec::{BootstrapCommand, ExecutionContext, command};
use crate::utils::helpers::{self, LldThreads, add_dylib_path, exe, libdir, linker_args, t};
use crate::utils::tracing::format_location;

mod cargo;
mod cli_paths;
mod step_stack;
#[cfg(test)]
mod tests;

/// Builds and performs different [`Self::kind`]s of stuff and actions, taking
/// into account build configuration from e.g. bootstrap.toml.
pub(crate) struct Builder<'a> {
    /// Build configuration from e.g. bootstrap.toml.
    pub sess: &'a Session,

    /// The stage to use. Either implicitly determined based on subcommand, or
    /// explicitly specified with `--stage N`. Normally this is the stage we
    /// use, but sometimes we want to run steps with a lower stage than this.
    pub top_stage: u32,

    /// What to build or what action to perform.
    pub kind: Kind,

    /// A cache of outputs of [`Step`]s so we can avoid running steps we already
    /// ran.
    cache: Cache,

    /// A stack of [`Step`]s to run before we can run this builder. The output
    /// of steps is cached in [`Self::cache`].
    stack: RefCell<Vec<Box<dyn AnyDebug>>>,

    /// The total amount of time we spent running [`Step`]s in [`Self::stack`].
    time_spent_on_dependencies: Cell<Duration>,

    /// The paths passed on the command line. Used by steps to figure out what
    /// to do. For example: with `./x check foo bar` we get `paths=["foo",
    /// "bar"]`.
    pub paths: Vec<PathBuf>,

    /// Cached list of submodules from self.sess.src.
    submodule_paths_cache: OnceLock<Vec<String>>,

    /// When enabled by tests, this causes the top-level steps that _would_ be
    /// executed to be logged instead. Used by snapshot tests of command-line
    /// paths-to-steps handling.
    #[expect(clippy::type_complexity)]
    log_cli_step_for_tests:
        Option<Box<dyn Fn(&CommandLineStepDescription, &[PathSet], &[TargetSelection])>>,
}

impl Deref for Builder<'_> {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        self.sess
    }
}

/// This trait is similar to `Any`, except that it also exposes the underlying
/// type's [`Debug`] implementation.
///
/// (Trying to debug-print `dyn Any` results in the unhelpful `"Any { .. }"`.)
pub trait AnyDebug: Any + Debug {}
impl<T: Any + Debug> AnyDebug for T {}
impl dyn AnyDebug {
    /// Equivalent to `<dyn Any>::downcast_ref`.
    fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }

    // Feel free to add other `dyn Any` methods as necessary.
}

/// A unit of work within bootstrap that is cached to avoid redundant execution.
/// Steps can be performed via [`Builder::ensure`].
///
/// Historically, steps also participated in command-line processing.
/// That responsibility has been split off into the larger [`CommandLineStep`] trait,
/// which helper steps don't need to implement.
pub(crate) trait Step: 'static + Clone + Debug + PartialEq + Eq + Hash {
    /// Result type of [`Step::run`]. Stored in the step cache for later lookup.
    type Output: Clone;

    /// Executes this step.
    ///
    /// Called by [`Builder::ensure`] if no cached result was found for this step.
    fn run(self, builder: &Builder<'_>) -> Self::Output;

    /// Returns metadata of the step, for tests.
    #[cfg_attr(not(any(test, feature = "tracing")), expect(dead_code))]
    fn metadata(&self) -> Option<StepMetadata> {
        None
    }
}

/// Every [`CommandLineStep`] is also a [`Step`].
impl<S: CommandLineStep> Step for S {
    type Output = <S as CommandLineStep>::Output;

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        <S as CommandLineStep>::run(self, builder)
    }

    fn metadata(&self) -> Option<StepMetadata> {
        <S as CommandLineStep>::metadata(self)
    }
}

/// A [`Step`] that can be selected by command-line arguments.
///
/// A blanket impl allows every [`CommandLineStep`] to be used as a [`Step`].
/// This is arguably nicer than having it be a subtrait, because it avoids the
/// need for two separate `impl` blocks per command-line-step type.
pub(crate) trait CommandLineStep: 'static + Clone + Debug + PartialEq + Eq + Hash {
    /// Result type of [`Step::run`].
    type Output: Clone;

    /// If this value is true, then the values of `run.target` passed to the `make_run` function of
    /// this Step will be determined based on the `--host` flag.
    /// If this value is false, then they will be determined based on the `--target` flag.
    ///
    /// A corollary of the above is that if this is set to true, then the step will be skipped if
    /// `--target` was specified, but `--host` was explicitly set to '' (empty string).
    const IS_HOST: bool = false;

    /// Called to allow steps to register the command-line paths that should
    /// cause them to run.
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_>;

    /// Should this step run when the user invokes bootstrap with a subcommand
    /// but no paths/aliases?
    ///
    /// For example, `./x test` runs all default test steps, and `./x dist`
    /// runs all default dist steps.
    ///
    /// Most steps are always default or always non-default, and just return
    /// true or false. But some steps are conditionally default, based on
    /// bootstrap config or the availability of ambient tools.
    ///
    /// If the underlying check should not be performed repeatedly
    /// (e.g. because it probes command-line tools),
    /// consider memoizing its outcome via a field in the builder.
    fn is_default_step(_builder: &Builder<'_>) -> bool {
        false
    }

    /// Called directly by the bootstrap `Step` handler when not triggered indirectly by other `Step`s using [`Builder::ensure`].
    /// For example, `./x.py test bootstrap` runs this for `test::Bootstrap`. Similarly, `./x.py test` runs it for every step
    /// that is listed by the `describe` macro in [`Builder::get_step_descriptions`].
    fn make_run(_run: RunConfig<'_>);

    /// Used as the implementation of [`Step::run`].
    fn run(self, builder: &Builder<'_>) -> Self::Output;

    /// Used as the implementation of [`Step::metadata`].
    fn metadata(&self) -> Option<StepMetadata> {
        None
    }
}

/// Metadata that describes an executed step, mostly for testing and tracing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StepMetadata {
    name: String,
    kind: Kind,
    target: TargetSelection,
    built_by: Option<Compiler>,
    stage: Option<u32>,
    /// Additional opaque string printed in the metadata
    metadata: Option<String>,
}

impl StepMetadata {
    pub fn build(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Build)
    }

    pub fn check(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Check)
    }

    pub fn clippy(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Clippy)
    }

    pub fn doc(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Doc)
    }

    pub fn dist(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Dist)
    }

    pub fn test(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Test)
    }

    pub fn run(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, Kind::Run)
    }

    pub fn new(name: &str, target: TargetSelection, kind: Kind) -> Self {
        Self { name: name.to_string(), kind, target, built_by: None, stage: None, metadata: None }
    }

    pub fn built_by(mut self, compiler: Compiler) -> Self {
        self.built_by = Some(compiler);
        self
    }

    pub fn stage(mut self, stage: u32) -> Self {
        self.stage = Some(stage);
        self
    }

    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }

    #[cfg_attr(not(any(test, feature = "tracing")), expect(dead_code))]
    pub(crate) fn get_stage(&self) -> Option<u32> {
        self.stage.or(self
            .built_by
            // For std, its stage corresponds to the stage of the compiler that builds it.
            // For everything else, a stage N things gets built by a stage N-1 compiler.
            .map(|compiler| if self.name == "std" { compiler.stage } else { compiler.stage + 1 }))
    }

    #[cfg_attr(not(feature = "tracing"), expect(dead_code))]
    pub(crate) fn get_name(&self) -> &str {
        &self.name
    }

    #[cfg_attr(not(feature = "tracing"), expect(dead_code))]
    pub(crate) fn get_target(&self) -> TargetSelection {
        self.target
    }
}

pub struct RunConfig<'a> {
    pub builder: &'a Builder<'a>,
    pub target: TargetSelection,
    pub paths: Vec<PathSet>,
}

impl RunConfig<'_> {
    pub fn build_triple(&self) -> TargetSelection {
        self.builder.sess.host_target
    }

    /// Return a list of crate names selected by `run.paths`.
    #[track_caller]
    pub fn cargo_crates_in_set(&self) -> Vec<String> {
        let mut crates = Vec::new();
        for krate in &self.paths {
            let path = &krate.assert_single_path().path;

            let crate_name = self
                .builder
                .crate_paths
                .get(path)
                .unwrap_or_else(|| panic!("missing crate for path {}", path.display()));

            crates.push(crate_name.to_string());
        }
        crates
    }

    /// Given an `alias` selected by the `Step` and the paths passed on the command line,
    /// return a list of the crates that should be built.
    ///
    /// Normally, people will pass *just* `library` if they pass it.
    /// But it's possible (although strange) to pass something like `library std core`.
    /// Build all crates anyway, as if they hadn't passed the other args.
    pub fn make_run_crates(&self, alias: Alias) -> Vec<String> {
        let has_alias =
            self.paths.iter().any(|set| set.assert_single_path().path.ends_with(alias.as_str()));
        if !has_alias {
            return self.cargo_crates_in_set();
        }

        let crates = match alias {
            Alias::Library => self.builder.in_tree_crates("sysroot", Some(self.target)),
            Alias::Compiler => self.builder.in_tree_crates("rustc-main", Some(self.target)),
        };

        crates.into_iter().map(|krate| krate.name.to_string()).collect()
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Alias {
    Library,
    Compiler,
}

impl Alias {
    fn as_str(self) -> &'static str {
        match self {
            Alias::Library => "library",
            Alias::Compiler => "compiler",
        }
    }
}

/// A description of the crates in this set, suitable for passing to `builder.info`.
///
/// `crates` should be generated by [`RunConfig::cargo_crates_in_set`].
pub fn crate_description(crates: &[impl AsRef<str>]) -> String {
    if crates.is_empty() {
        return "".into();
    }

    let mut descr = String::from("{");
    descr.push_str(crates[0].as_ref());
    for krate in &crates[1..] {
        descr.push_str(", ");
        descr.push_str(krate.as_ref());
    }
    descr.push('}');
    descr
}

struct CommandLineStepDescription {
    is_host: bool,
    should_run: fn(ShouldRun<'_>) -> ShouldRun<'_>,
    is_default_step_fn: fn(&Builder<'_>) -> bool,
    make_run: fn(RunConfig<'_>),
    name: &'static str,

    /// Kind that was passed to [`CommandLineStepDescription::from`].
    #[cfg_attr(not(test), expect(dead_code, reason = "currently only needed by tests"))]
    kind: Kind,
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct TaskPath {
    pub path: PathBuf,
}

impl Debug for TaskPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path.display())
    }
}

/// Collection of paths used to match a task rule.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum PathSet {
    /// A collection of individual paths or aliases.
    ///
    /// These are generally matched as a path suffix. For example, a
    /// command-line value of `std` will match if `library/std` is in the
    /// set.
    ///
    /// NOTE: the paths within a set should all select the same unit of work.
    /// For example, `src/librustdoc` and `src/tools/rustdoc` should be in the same set,
    /// but `library/core` and `library/std` generally should not, unless there's no way (for that Step)
    /// to build them separately.
    Set(BTreeSet<TaskPath>),
    /// A "suite" of paths.
    ///
    /// These can match as a path suffix (like `Set`), or as a prefix. For
    /// example, a command-line value of `tests/ui/abi/variadic-ffi.rs`
    /// will match `tests/ui`. A command-line value of `ui` would also
    /// match `tests/ui`.
    Suite(TaskPath),
}

impl PathSet {
    fn one<P: Into<PathBuf>>(path: P) -> PathSet {
        let mut set = BTreeSet::new();
        set.insert(TaskPath { path: path.into() });
        PathSet::Set(set)
    }

    fn has(&self, needle: &Path) -> bool {
        match self {
            PathSet::Set(set) => set.iter().any(|p| Self::check(p, needle)),
            PathSet::Suite(suite) => Self::check(suite, needle),
        }
    }

    // internal use only
    fn check(p: &TaskPath, needle: &Path) -> bool {
        // This order is important for retro-compatibility, as `starts_with` was introduced later.
        p.path.ends_with(needle) || p.path.starts_with(needle)
    }

    /// A convenience wrapper for Steps which know they have no aliases and all their sets contain only a single path.
    ///
    /// This can be used with [`ShouldRun::crate_or_deps`], [`ShouldRun::path`], or [`ShouldRun::alias`].
    #[track_caller]
    pub fn assert_single_path(&self) -> &TaskPath {
        match self {
            PathSet::Set(set) => {
                assert_eq!(set.len(), 1, "called assert_single_path on multiple paths");
                set.iter().next().unwrap()
            }
            PathSet::Suite(_) => unreachable!("called assert_single_path on a Suite path"),
        }
    }
}

impl CommandLineStepDescription {
    fn from<S: CommandLineStep>(kind: Kind) -> CommandLineStepDescription {
        CommandLineStepDescription {
            is_host: S::IS_HOST,
            should_run: S::should_run,
            is_default_step_fn: S::is_default_step,
            make_run: S::make_run,
            name: std::any::type_name::<S>(),
            kind,
        }
    }

    fn maybe_run(&self, builder: &Builder<'_>, mut pathsets: Vec<PathSet>) {
        pathsets.retain(|set| !self.is_excluded(builder, set));

        if pathsets.is_empty() {
            return;
        }

        // Determine the targets participating in this rule.
        let targets = if self.is_host { &builder.hosts } else { &builder.targets };

        // Log the step that's about to run, for snapshot tests.
        if let Some(ref log_cli_step) = builder.log_cli_step_for_tests {
            log_cli_step(self, &pathsets, targets);
            // Return so that the step won't actually run in snapshot tests.
            return;
        }

        for target in targets {
            let run = RunConfig { builder, paths: pathsets.clone(), target: *target };
            (self.make_run)(run);
        }
    }

    fn is_excluded(&self, builder: &Builder<'_>, pathset: &PathSet) -> bool {
        if builder.config.skip.iter().any(|e| pathset.has(e)) {
            if !matches!(builder.config.get_dry_run(), DryRun::SelfCheck) {
                println!("Skipping {pathset:?} because it is excluded");
            }
            return true;
        }

        if !builder.config.skip.is_empty()
            && !matches!(builder.config.get_dry_run(), DryRun::SelfCheck)
        {
            builder.do_if_verbose(|| {
                println!(
                    "{:?} not skipped for {:?} -- not in {:?}",
                    pathset, self.name, builder.config.skip
                )
            });
        }
        false
    }
}

/// Builder that allows steps to register command-line paths/aliases that
/// should cause those steps to be run.
///
/// For example, if the user invokes `./x test compiler` or `./x doc unstable-book`,
/// this allows bootstrap to determine what steps "compiler" or "unstable-book"
/// correspond to.
pub struct ShouldRun<'a> {
    pub builder: &'a Builder<'a>,

    // use a BTreeSet to maintain sort order
    paths: BTreeSet<PathSet>,
}

impl<'a> ShouldRun<'a> {
    fn new(builder: &'a Builder<'_>) -> ShouldRun<'a> {
        ShouldRun { builder, paths: BTreeSet::new() }
    }

    /// The corresponding step should run if the bootstrap command-line selects
    /// the given crate or any of its (local) dependencies.
    ///
    /// Delegates to [`Self::crate_or_deps_filtered`] with a filter that accepts all crates.
    pub(crate) fn crate_or_deps(self, root_crate_name: &str) -> Self {
        self.crate_or_deps_filtered(root_crate_name, |_: &Crate| true)
    }

    /// The corresponding step should run if the bootstrap command-line selects
    /// the given crate or any of its (local) dependencies, not counting any
    /// crates rejected by the given filter function.
    ///
    /// `make_run` will be called a single time with all matching command-line paths.
    pub(crate) fn crate_or_deps_filtered(
        mut self,
        root_crate_name: &str,
        crate_filter_fn: impl Fn(&Crate) -> bool,
    ) -> Self {
        let crates = self.builder.in_tree_crates(root_crate_name, None);
        for krate in crates {
            if !crate_filter_fn(krate) {
                continue;
            }

            let path = krate.local_path(self.builder);
            self.paths.insert(PathSet::one(path));
        }
        self
    }

    // single alias, which does not correspond to any on-disk path
    pub fn alias(self, alias: &str) -> Self {
        self.assert_valid_alias(alias);
        self.alias_without_assert(alias)
    }

    /// Like [`Self::alias`], but does not assert the absence of a path with the same name.
    ///
    /// Needed by [`setup::Profile`], which registers aliases named `compiler` and `library`
    /// that happen to coincide with directory names.
    pub fn alias_without_assert(mut self, alias: &str) -> Self {
        self.paths.insert(PathSet::Set(iter::once(TaskPath { path: alias.into() }).collect()));
        self
    }

    fn assert_valid_alias(&self, alias: &str) {
        assert!(
            !self.builder.src.join(alias).exists(),
            "use `builder.path()` for real paths: {alias}"
        );
    }

    fn assert_valid_path(&self, path: &str) {
        let submodules_paths = self.builder.submodule_paths();

        // assert only if `p` isn't submodule
        if !submodules_paths.iter().any(|sm_p| path.contains(sm_p)) {
            assert!(
                self.builder.src.join(path).exists(),
                "`should_run.path` should correspond to a real on-disk path - use `alias` if there is no relevant path: {path}"
            );
        }
    }

    /// A single path
    ///
    /// Must be an on-disk path; use [`alias`][Self::alias] for names that do not
    /// correspond to on-disk paths.
    pub fn path(mut self, path: &str) -> Self {
        self.assert_valid_path(path);

        let task = TaskPath { path: path.into() };
        self.paths.insert(PathSet::Set(BTreeSet::from_iter([task])));
        self
    }

    /// Registers a path, and an alias that is treated as equivalent to that path.
    pub fn path_with_alias(mut self, path: &str, alias: &str) -> Self {
        self.assert_valid_path(path);
        self.assert_valid_alias(alias);

        let set = [path, alias]
            .into_iter()
            .map(|p| TaskPath { path: PathBuf::from(p) })
            .collect::<BTreeSet<_>>();
        self.paths.insert(PathSet::Set(set));
        self
    }

    /// Multiple on-disk paths that should select the same unit of work.
    pub fn multi_path(mut self, paths: &[&str]) -> Self {
        let mut set = BTreeSet::new();
        for path in paths {
            self.assert_valid_path(path);
            set.insert(TaskPath { path: (*path).into() });
        }
        self.paths.insert(PathSet::Set(set));
        self
    }

    pub fn suite_path(mut self, suite: &str) -> Self {
        self.paths.insert(PathSet::Suite(TaskPath { path: suite.into() }));
        self
    }

    /// When the corresponding step is run "by default" (without explicit command-line paths),
    /// act as though the user had explicitly specified these paths.
    fn default_pathsets(&self) -> Vec<PathSet> {
        self.paths.iter().cloned().collect::<Vec<_>>()
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, PartialOrd, Ord, ValueEnum)]
pub enum Kind {
    #[value(alias = "b")]
    Build,
    #[value(alias = "c")]
    Check,
    Clippy,
    Fix,
    Format,
    #[value(alias = "t")]
    Test,
    Miri,
    MiriSetup,
    MiriTest,
    Bench,
    #[value(alias = "d")]
    Doc,
    Clean,
    Dist,
    Install,
    #[value(alias = "r")]
    Run,
    Setup,
    Vendor,
    Perf,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Build => "build",
            Kind::Check => "check",
            Kind::Clippy => "clippy",
            Kind::Fix => "fix",
            Kind::Format => "fmt",
            Kind::Test => "test",
            Kind::Miri => "miri",
            Kind::MiriSetup => panic!("`as_str` is not supported for `Kind::MiriSetup`."),
            Kind::MiriTest => panic!("`as_str` is not supported for `Kind::MiriTest`."),
            Kind::Bench => "bench",
            Kind::Doc => "doc",
            Kind::Clean => "clean",
            Kind::Dist => "dist",
            Kind::Install => "install",
            Kind::Run => "run",
            Kind::Setup => "setup",
            Kind::Vendor => "vendor",
            Kind::Perf => "perf",
        }
    }

    pub fn description(&self) -> String {
        match self {
            Kind::Test => "Testing",
            Kind::Bench => "Benchmarking",
            Kind::Doc => "Documenting",
            Kind::Run => "Running",
            Kind::Clippy => "Linting",
            Kind::Perf => "Profiling & benchmarking",
            _ => {
                let title_letter = self.as_str()[0..1].to_ascii_uppercase();
                return format!("{title_letter}{}ing", &self.as_str()[1..]);
            }
        }
        .to_owned()
    }

    /// Is this a command similar to check, which only runs the compiler frontend and doesn't
    /// build code for the target? (it can still build code for the host, i.e. proc macros).
    pub fn is_check_like(&self) -> bool {
        match self {
            Kind::Check | Kind::Clippy | Kind::Fix | Kind::Doc => true,
            Kind::Build
            | Kind::Format
            | Kind::Test
            | Kind::Miri
            | Kind::MiriSetup
            | Kind::MiriTest
            | Kind::Bench
            | Kind::Clean
            | Kind::Dist
            | Kind::Install
            | Kind::Run
            | Kind::Setup
            | Kind::Vendor
            | Kind::Perf => false,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Libdir {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for Libdir {
    type Output = PathBuf;

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let relative_sysroot_libdir = builder.sysroot_libdir_relative(self.compiler);
        let sysroot = builder.sysroot(self.compiler).join(relative_sysroot_libdir).join("rustlib");

        if !builder.config.dry_run() {
            // Avoid deleting the `rustlib/` directory we just copied (in `impl CommandLineStep for
            // Sysroot`).
            if !builder.download_rustc() {
                let sysroot_target_libdir = sysroot.join(self.target).join("lib");
                builder.do_if_verbose(|| {
                    eprintln!(
                        "Removing sysroot {} to avoid caching bugs",
                        sysroot_target_libdir.display()
                    )
                });
                let _ = fs::remove_dir_all(&sysroot_target_libdir);
                t!(fs::create_dir_all(&sysroot_target_libdir));
            }

            if self.compiler.stage == 0 {
                // The stage 0 compiler for the build triple is always pre-built. Ensure that
                // `libLLVM.so` ends up in the target libdir, so that ui-fulldeps tests can use
                // it when run.
                dist::maybe_install_llvm_target(
                    builder,
                    self.compiler.host,
                    &builder.sysroot(self.compiler),
                );
            }
        }

        sysroot
    }
}

#[cfg(feature = "tracing")]
pub const STEP_SPAN_TARGET: &str = "STEP";

impl<'a> Builder<'a> {
    fn get_step_descriptions(kind: Kind) -> Vec<CommandLineStepDescription> {
        macro_rules! describe {
            ($($rule:ty),+ $(,)?) => {{
                vec![$(CommandLineStepDescription::from::<$rule>(kind)),+]
            }};
        }
        match kind {
            Kind::Build => describe!(
                compile::Std,
                compile::Rustc,
                compile::Assemble,
                compile::CraneliftCodegenBackend,
                compile::GccCodegenBackend,
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
                tool::FeaturesStatusDump,
                tool::Cargo,
                tool::RustAnalyzer,
                tool::RustAnalyzerProcMacroSrv,
                tool::Rustdoc,
                tool::Clippy,
                tool::CargoClippy,
                llvm::Llvm,
                gcc::Gcc,
                llvm::Sanitizers,
                tool::Rustfmt,
                tool::Cargofmt,
                tool::Miri,
                tool::CargoMiri,
                llvm::Lld,
                llvm::Enzyme,
                llvm::RustOffload,
                llvm::CrtBeginEnd,
                tool::RustdocGUITest,
                tool::OptimizedDist,
                tool::CoverageDump,
                tool::LlvmBitcodeLinker,
                tool::RustcPerf,
                tool::WasmComponentLd,
                tool::LldWrapper
            ),
            Kind::Clippy => describe!(
                clippy::Std,
                clippy::Rustc,
                clippy::Bootstrap,
                clippy::BuildHelper,
                clippy::BuildManifest,
                clippy::CargoMiri,
                clippy::Clippy,
                clippy::CodegenGcc,
                clippy::CollectLicenseMetadata,
                clippy::Compiletest,
                clippy::CoverageDump,
                clippy::Jsondocck,
                clippy::Jsondoclint,
                clippy::LintDocs,
                clippy::LlvmBitcodeLinker,
                clippy::Miri,
                clippy::MiroptTestTools,
                clippy::OptDist,
                clippy::RemoteTestClient,
                clippy::RemoteTestServer,
                clippy::RustAnalyzer,
                clippy::Rustdoc,
                clippy::Rustfmt,
                clippy::RustInstaller,
                clippy::TestFloatParse,
                clippy::Tidy,
                clippy::CI,
            ),
            Kind::Check | Kind::Fix => describe!(
                check::Rustc,
                check::Rustdoc,
                check::CraneliftCodegenBackend,
                check::GccCodegenBackend,
                check::Clippy,
                check::Miri,
                check::CargoMiri,
                check::Priroda,
                check::MiroptTestTools,
                check::Rustfmt,
                check::RustAnalyzer,
                check::TestFloatParse,
                check::Bootstrap,
                check::RunMakeSupport,
                check::Compiletest,
                check::RustdocGuiTest,
                check::FeaturesStatusDump,
                check::CoverageDump,
                check::Linkchecker,
                check::BumpStage0,
                check::Tidy,
                // This has special staging logic, it may run on stage 1 while others run on stage 0.
                // It takes quite some time to build stage 1, so put this at the end.
                //
                // FIXME: This also helps bootstrap to not interfere with stage 0 builds. We should probably fix
                // that issue somewhere else, but we still want to keep `check::Std` at the end so that the
                // quicker steps run before this.
                check::Std,
            ),
            Kind::Test => describe!(
                crate::core::build_steps::toolstate::ToolStateCheck,
                test::Tidy,
                test::BootstrapPy,
                test::Bootstrap,
                test::Ui,
                test::Crashes,
                test::Coverage,
                test::CoverageModeAlias,
                test::MirOpt,
                test::CodegenLlvm,
                test::CodegenUnits,
                test::AssemblyLlvm,
                test::Incremental,
                test::Debuginfo,
                test::UiFullDeps,
                test::RustdocHtml,
                test::CoverageRunRustdoc,
                test::Pretty,
                test::CodegenCranelift,
                test::CodegenGCC,
                test::Crate,
                test::CrateLibrustc,
                test::CrateRustdoc,
                test::CrateRustdocJsonTypes,
                test::CrateBootstrap,
                test::RemoteTestClientTests,
                test::Linkcheck,
                test::TierCheck,
                test::Cargotest,
                test::Cargo,
                test::RustAnalyzer,
                test::ErrorIndex,
                test::Distcheck,
                test::Nomicon,
                test::Reference,
                test::RustdocBook,
                test::RustByExample,
                test::TheBook,
                test::UnstableBook,
                test::RustcBook,
                test::LintDocs,
                test::EmbeddedBook,
                test::EditionGuide,
                test::Rustfmt,
                test::Miri,
                test::CargoMiri,
                test::Priroda,
                test::Clippy,
                test::CompiletestTest,
                test::StdarchVerify,
                test::CrateRunMakeSupport,
                test::CrateBuildHelper,
                test::RustdocJSStd,
                test::RustdocJSNotStd,
                test::RustdocGUI,
                test::RustdocTheme,
                test::RustdocUi,
                test::RustdocJson,
                test::HtmlCheck,
                test::RustInstaller,
                test::TestFloatParse,
                test::CollectLicenseMetadata,
                test::RunMake,
                test::RunMakeCargo,
                test::BuildStd,
                test::StdSemverCheck,
                test::IntrinsicTest,
            ),
            Kind::Miri => describe!(test::Crate),
            Kind::Bench => describe!(test::Crate, test::CrateLibrustc, test::CrateRustdoc),
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
                doc::Cargo,
                doc::CargoBook,
                doc::Clippy,
                doc::ClippyBook,
                doc::CompilerWithTools,
                doc::Miri,
                doc::EmbeddedBook,
                doc::EditionGuide,
                doc::StyleGuide,
                doc::Tidy,
                doc::Bootstrap,
                doc::Releases,
                doc::RunMakeSupport,
                doc::BuildHelper,
                doc::Compiletest,
            ),
            Kind::Dist => describe!(
                dist::Docs,
                dist::RustcDocs,
                dist::JsonDocs,
                dist::Mingw,
                dist::Rustc,
                dist::CraneliftCodegenBackend,
                dist::GccCodegenBackend,
                dist::Std,
                dist::RustcDev,
                dist::Analysis,
                dist::Src,
                dist::Cargo,
                dist::RustAnalyzer,
                dist::Rustfmt,
                dist::Clippy,
                dist::Miri,
                dist::LlvmTools,
                dist::LlvmBitcodeLinker,
                dist::RustDev,
                dist::Enzyme,
                dist::Offload,
                dist::Bootstrap,
                dist::Extended,
                // It seems that PlainSourceTarball somehow changes how some of the tools
                // perceive their dependencies (see #93033) which would invalidate fingerprints
                // and force us to rebuild tools after vendoring dependencies.
                // To work around this, create the Tarball after building all the tools.
                dist::PlainSourceTarball,
                dist::PlainSourceTarballGpl,
                dist::BuildManifest,
                dist::ReproducibleArtifacts,
                dist::GccDev,
                dist::Gcc
            ),
            Kind::Install => describe!(
                install::Docs,
                install::Std,
                // During the Rust compiler (rustc) installation process, we copy the entire sysroot binary
                // path (build/host/stage2/bin). Since the building tools also make their copy in the sysroot
                // binary path, we must install rustc before the tools. Otherwise, the rust-installer will
                // install the same binaries twice for each tool, leaving backup files (*.old) as a result.
                install::Rustc,
                install::RustcDev,
                install::Cargo,
                install::RustAnalyzer,
                install::Rustfmt,
                install::Clippy,
                install::Miri,
                install::LlvmTools,
                install::Src,
                install::RustcCodegenCranelift,
                install::LlvmBitcodeLinker
            ),
            Kind::Run => describe!(
                run::BuildManifest,
                run::BumpStage0,
                run::ReplaceVersionPlaceholder,
                run::Miri,
                run::CollectLicenseMetadata,
                run::GenerateCopyright,
                run::GenerateWindowsSys,
                run::GenerateCompletions,
                run::UnicodeTableGenerator,
                run::FeaturesStatusDump,
                run::CyclicStep,
                run::CoverageDump,
                run::Rustfmt,
                run::GenerateHelp,
            ),
            Kind::Setup => {
                describe!(setup::Profile, setup::Hook, setup::Link, setup::Editor)
            }
            Kind::Clean => describe!(clean::CleanAll, clean::Rustc, clean::Std),
            Kind::Vendor => describe!(vendor::Vendor),
            // special-cased in Session::build()
            Kind::Format | Kind::Perf => vec![],
            Kind::MiriTest | Kind::MiriSetup => unreachable!(),
        }
    }

    pub fn get_help(sess: &Session, kind: Kind) -> Option<String> {
        let step_descriptions = Builder::get_step_descriptions(kind);
        if step_descriptions.is_empty() {
            return None;
        }

        let builder = Self::new_internal(sess, kind, vec![]);
        let builder = &builder;

        let mut should_run = ShouldRun::new(builder);
        for desc in step_descriptions {
            should_run = (desc.should_run)(should_run);
        }
        let mut help = String::from("Available paths:\n");
        let mut add_path = |path: &Path| {
            t!(write!(help, "    ./x.py {} {}\n", kind.as_str(), path.display()));
        };
        for pathset in should_run.paths {
            match pathset {
                PathSet::Set(set) => {
                    for path in set {
                        add_path(&path.path);
                    }
                }
                PathSet::Suite(path) => {
                    add_path(&path.path.join("..."));
                }
            }
        }
        Some(help)
    }

    fn new_internal(sess: &Session, kind: Kind, paths: Vec<PathBuf>) -> Builder<'_> {
        Builder {
            sess,
            top_stage: sess.config.stage,
            kind,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
            time_spent_on_dependencies: Cell::new(Duration::new(0, 0)),
            paths,
            submodule_paths_cache: Default::default(),
            log_cli_step_for_tests: None,
        }
    }

    pub fn new(sess: &Session) -> Builder<'_> {
        let paths = &sess.config.paths;
        let (kind, paths) = match sess.config.cmd {
            Subcommand::Build { .. } => (Kind::Build, &paths[..]),
            Subcommand::Check { .. } => (Kind::Check, &paths[..]),
            Subcommand::Clippy { .. } => (Kind::Clippy, &paths[..]),
            Subcommand::Fix { .. } => (Kind::Fix, &paths[..]),
            Subcommand::Doc { .. } => (Kind::Doc, &paths[..]),
            Subcommand::Test { .. } => (Kind::Test, &paths[..]),
            Subcommand::Miri { .. } => (Kind::Miri, &paths[..]),
            Subcommand::Bench { .. } => (Kind::Bench, &paths[..]),
            Subcommand::Dist => (Kind::Dist, &paths[..]),
            Subcommand::Install => (Kind::Install, &paths[..]),
            Subcommand::Run { .. } => (Kind::Run, &paths[..]),
            Subcommand::Clean { .. } => (Kind::Clean, &paths[..]),
            Subcommand::Format { .. } => (Kind::Format, &[][..]),
            Subcommand::Setup { profile: ref path } => (
                Kind::Setup,
                path.as_ref().map_or([].as_slice(), |path| std::slice::from_ref(path)),
            ),
            Subcommand::Vendor { .. } => (Kind::Vendor, &paths[..]),
            Subcommand::Perf { .. } => (Kind::Perf, &paths[..]),
        };

        StepStack::with_current(|stack| stack.clear());
        Self::new_internal(sess, kind, paths.to_owned())
    }

    pub fn execute_cli(&self) {
        self.run_step_descriptions(&Builder::get_step_descriptions(self.kind), &self.paths);
    }

    /// Run all default documentation steps to build documentation.
    pub fn run_default_doc_steps(&self) {
        // It's important that we don't just call `run_step_descriptions` here,
        // because that would cause `--skip` handling for actual command-line
        // arguments to inappropriately skip these steps.
        //
        // This function is nevertheless a bit of a hack, to work around the
        // fact that we don't have a good way to simulate `./x doc` without
        // also simulating parts of command-line selector handling.

        for desc in &Builder::get_step_descriptions(Kind::Doc) {
            if !(desc.is_default_step_fn)(self) {
                continue;
            }

            let should_run = (desc.should_run)(ShouldRun::new(self));
            let default_pathsets = should_run.default_pathsets();

            let targets = if desc.is_host { &self.hosts } else { &self.targets };
            for &target in targets {
                let run = RunConfig { builder: self, target, paths: default_pathsets.clone() };
                (desc.make_run)(run);
            }
        }
    }

    pub fn doc_rust_lang_org_channel(&self) -> String {
        let channel = match &*self.config.channel {
            "stable" => &self.version,
            "beta" => "beta",
            "nightly" | "dev" => "nightly",
            // custom build of rustdoc maybe? link to the latest stable docs just in case
            _ => "stable",
        };

        format!("https://doc.rust-lang.org/{channel}")
    }

    fn run_step_descriptions(&self, v: &[CommandLineStepDescription], paths: &[PathBuf]) {
        cli_paths::match_paths_to_steps_and_run(self, v, paths);
    }

    /// Obtain a compiler at a given stage and for a given host (i.e., this is the target that the
    /// compiler will run on, *not* the target it will build code for). Explicitly does not take
    /// `Compiler` since all `Compiler` instances are meant to be obtained through this function,
    /// since it ensures that they are valid (i.e., built and assembled).
    #[track_caller]
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "Builder::compiler",
            target = "COMPILER",
            skip_all,
            fields(
                stage = stage,
                host = ?host,
            ),
        ),
    )]
    pub fn compiler(&self, stage: u32, host: TargetSelection) -> Compiler {
        self.ensure(compile::Assemble { target_compiler: Compiler::new(stage, host) })
    }

    /// This function can be used to provide a build compiler for building
    /// the standard library, in order to avoid unnecessary rustc builds in case where std uplifting
    /// would happen anyway.
    ///
    /// This is an important optimization mainly for CI.
    ///
    /// Normally, to build stage N libstd, we need stage N rustc.
    /// However, if we know that we will uplift libstd from stage 1 anyway, building the stage N
    /// rustc can be wasteful.
    /// In particular, if we do a cross-compiling dist stage 2 build from target1 to target2,
    /// we need:
    /// - stage 2 libstd for target2 (uplifted from stage 1, where it was built by target1 rustc)
    /// - stage 2 rustc for target2
    ///
    /// However, without this optimization, we would also build stage 2 rustc for **target1**,
    /// which is completely wasteful.
    #[track_caller]
    pub fn compiler_for_std(&self, stage: u32) -> Compiler {
        if compile::Std::should_be_uplifted_from_stage_1(self, stage) {
            self.compiler(1, self.host_target)
        } else {
            self.compiler(stage, self.host_target)
        }
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
    /// See `force_use_stage1` and `force_use_stage2` for documentation on what each argument is.
    #[track_caller]
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "Builder::compiler_for",
            target = "COMPILER_FOR",
            skip_all,
            fields(
                stage = stage,
                host = ?host,
                target = ?target,
            ),
        ),
    )]
    /// FIXME: This function is unnecessary (and dangerous, see <https://github.com/rust-lang/rust/issues/137469>).
    /// We already have uplifting logic for the compiler, so remove this.
    pub fn compiler_for(
        &self,
        stage: u32,
        host: TargetSelection,
        target: TargetSelection,
    ) -> Compiler {
        let mut resolved_compiler = if self.sess.force_use_stage2(stage) {
            trace!(target: "COMPILER_FOR", ?stage, "force_use_stage2");
            self.compiler(2, self.config.host_target)
        } else if self.sess.force_use_stage1(stage, target) {
            trace!(target: "COMPILER_FOR", ?stage, "force_use_stage1");
            self.compiler(1, self.config.host_target)
        } else {
            trace!(target: "COMPILER_FOR", ?stage, ?host, "no force, fallback to `compiler()`");
            self.compiler(stage, host)
        };

        if stage != resolved_compiler.stage {
            resolved_compiler.forced_compiler(true);
        }

        trace!(target: "COMPILER_FOR", ?resolved_compiler);
        resolved_compiler
    }

    /// Obtain a standard library for the given target that will be built by the passed compiler.
    /// The standard library will be linked to the sysroot of the passed compiler.
    ///
    /// Prefer using this method rather than manually invoking `Std::new`.
    ///
    /// Returns an optional build stamp, if libstd was indeed built.
    #[track_caller]
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "Builder::std",
            target = "STD",
            skip_all,
            fields(
                compiler = ?compiler,
                target = ?target,
            ),
        ),
    )]
    pub fn std(&self, compiler: Compiler, target: TargetSelection) -> Option<BuildStamp> {
        // FIXME: make the `Std` step return some type-level "proof" that std was indeed built,
        // and then require passing that to all Cargo invocations that we do.

        // The "stage 0" std is almost always precompiled and comes with the stage0 compiler, so we
        // have special logic for it, to avoid creating needless and confusing Std steps that don't
        // actually build anything.
        // We only allow building the stage0 stdlib if we do a local rebuild, so the stage0 compiler
        // actually comes from in-tree sources, and we're cross-compiling, so the stage0 for the
        // given `target` is not available.
        if compiler.stage == 0 {
            if target != compiler.host {
                if self.local_rebuild {
                    self.ensure(Std::new(compiler, target))
                } else {
                    panic!(
                        r"It is not possible to build the standard library for `{target}` using the stage0 compiler.
You have to build a stage1 compiler for `{}` first, and then use it to build a standard library for `{target}`.
Alternatively, you can set `build.local-rebuild=true` and use a stage0 compiler built from in-tree sources.
",
                        compiler.host
                    )
                }
            } else {
                // We still need to link the prebuilt standard library into the ephemeral stage0 sysroot
                self.ensure(StdLink::from_std(Std::new(compiler, target), compiler));
                None
            }
        } else {
            // This step both compiles the std and links it into the compiler's sysroot.
            // Yes, it's quite magical and side-effecty.. would be nice to refactor later.
            self.ensure(Std::new(compiler, target))
        }
    }

    #[track_caller]
    pub fn sysroot(&self, compiler: Compiler) -> PathBuf {
        self.ensure(compile::Sysroot::new(compiler))
    }

    /// Returns the bindir for a compiler's sysroot.
    #[track_caller]
    pub fn sysroot_target_bindir(&self, compiler: Compiler, target: TargetSelection) -> PathBuf {
        self.ensure(Libdir { compiler, target }).join(target).join("bin")
    }

    /// Returns the libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
    #[track_caller]
    pub fn sysroot_target_libdir(&self, compiler: Compiler, target: TargetSelection) -> PathBuf {
        self.ensure(Libdir { compiler, target }).join(target).join("lib")
    }

    pub fn sysroot_codegen_backends(&self, compiler: Compiler) -> PathBuf {
        self.sysroot_target_libdir(compiler, compiler.host).with_file_name("codegen-backends")
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
            libdir(self.config.host_target).as_ref()
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
            _ if compiler.stage == 0 => &self.sess.initial_relative_libdir,
            _ => Path::new("lib"),
        }
    }

    pub fn rustc_lib_paths(&self, compiler: Compiler) -> Vec<PathBuf> {
        let mut dylib_dirs = vec![self.rustc_libdir(compiler)];

        // Ensure that the downloaded LLVM libraries can be found.
        if self.config.llvm_ci_mode.download_from_ci() {
            let ci_llvm_lib = self.out.join(compiler.host).join("ci-llvm").join("lib");
            dylib_dirs.push(ci_llvm_lib);
        }

        dylib_dirs
    }

    /// Adds the compiler's directory of dynamic libraries to `cmd`'s dynamic
    /// library lookup path.
    pub fn add_rustc_lib_path(&self, compiler: Compiler, cmd: &mut BootstrapCommand) {
        // Windows doesn't need dylib path munging because the dlls for the
        // compiler live next to the compiler and the system will find them
        // automatically.
        if cfg!(any(windows, target_os = "cygwin")) {
            return;
        }

        add_dylib_path(self.rustc_lib_paths(compiler), cmd);
    }

    /// Gets a path to the compiler specified.
    pub fn rustc(&self, compiler: Compiler) -> PathBuf {
        if compiler.is_snapshot(self) {
            self.initial_rustc.clone()
        } else {
            self.sysroot(compiler).join("bin").join(exe("rustc", compiler.host))
        }
    }

    /// Gets a command to run the compiler specified, including the dynamic library
    /// path in case the executable has not been build with `rpath` enabled.
    pub fn rustc_cmd(&self, compiler: Compiler) -> BootstrapCommand {
        let mut cmd = command(self.rustc(compiler));
        self.add_rustc_lib_path(compiler, &mut cmd);
        cmd
    }

    /// Gets the paths to all of the compiler's codegen backends.
    fn codegen_backends(&self, compiler: Compiler) -> impl Iterator<Item = PathBuf> {
        fs::read_dir(self.sysroot_codegen_backends(compiler))
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|path| looks_like_codegen_backend(&path.path()))
            .map(|entry| entry.path())
    }

    /// Returns a path to `Rustdoc` that "belongs" to the `target_compiler`.
    /// It can be either a stage0 rustdoc or a locally built rustdoc that *links* to
    /// `target_compiler`.
    #[track_caller]
    pub fn rustdoc_for_compiler(&self, target_compiler: Compiler) -> PathBuf {
        self.ensure(tool::Rustdoc { target_compiler })
    }

    pub fn cargo_miri_cmd(&self, run_compiler: Compiler) -> BootstrapCommand {
        assert!(run_compiler.stage > 0, "miri can not be invoked at stage 0");

        let compilers = RustcPrivateCompilers::new(self, run_compiler.stage, self.sess.host_target);
        assert_eq!(run_compiler, compilers.target_compiler());

        // Prepare the tools
        let miri = self.ensure(tool::Miri::from_compilers(compilers));
        let cargo_miri = self.ensure(tool::CargoMiri::from_compilers(compilers));
        // Invoke cargo-miri, make sure it can find miri and cargo.
        let mut cmd = command(cargo_miri.tool_path);
        cmd.env("MIRI", &miri.tool_path);
        cmd.env("CARGO", &self.initial_cargo);
        // Need to add the `run_compiler` libs. Those are the libs produces *by* `build_compiler`
        // in `tool::ToolBuild` step, so they match the Miri we just built. However this means they
        // are actually living one stage up, i.e. we are running `stage1-tools-bin/miri` with the
        // libraries in `stage1/lib`. This is an unfortunate off-by-1 caused (possibly) by the fact
        // that Miri doesn't have an "assemble" step like rustc does that would cross the stage boundary.
        // We can't use `add_rustc_lib_path` as that's a NOP on Windows but we do need these libraries
        // added to the PATH due to the stage mismatch.
        // Also see https://github.com/rust-lang/rust/pull/123192#issuecomment-2028901503.
        add_dylib_path(self.rustc_lib_paths(run_compiler), &mut cmd);
        cmd
    }

    /// Create a Cargo command for running Clippy.
    /// The used Clippy is (or in the case of stage 0, already was) built using `build_compiler`.
    pub fn cargo_clippy_cmd(&self, build_compiler: Compiler) -> BootstrapCommand {
        if build_compiler.stage == 0 {
            let cargo_clippy =
                self.config.external_cargo_clippy.clone().unwrap_or_else(|| {
                    self.sess.config.download_clippy(&self.sess.initial_sysroot)
                });

            let mut cmd = command(cargo_clippy);
            cmd.env("CARGO", &self.initial_cargo);
            return cmd;
        }

        // If we're linting something with build_compiler stage N, we want to build Clippy stage N
        // and use that to lint it. That is why we use the `build_compiler` as the target compiler
        // for RustcPrivateCompilers. We will use build compiler stage N-1 to build Clippy stage N.
        let compilers = RustcPrivateCompilers::from_target_compiler(self, build_compiler);

        let _ = self.ensure(tool::Clippy::from_compilers(compilers));
        let cargo_clippy = self.ensure(tool::CargoClippy::from_compilers(compilers));
        let mut dylib_path = helpers::dylib_path();
        dylib_path.insert(0, self.sysroot(build_compiler).join("lib"));

        let mut cmd = command(cargo_clippy.tool_path);
        cmd.env(helpers::dylib_path_var(), env::join_paths(&dylib_path).unwrap());
        cmd.env("CARGO", &self.initial_cargo);
        cmd
    }

    pub fn rustdoc_cmd(&self, compiler: Compiler) -> BootstrapCommand {
        let mut cmd = command(self.bootstrap_out.join("rustdoc"));
        cmd.env("RUSTC_STAGE", compiler.stage.to_string())
            .env("RUSTC_SYSROOT", self.sysroot(compiler))
            // Note that this is *not* the sysroot_libdir because rustdoc must be linked
            // equivalently to rustc.
            .env("RUSTDOC_LIBDIR", self.rustc_libdir(compiler))
            .env("CFG_RELEASE_CHANNEL", &self.config.channel)
            .env("RUSTDOC_REAL", self.rustdoc_for_compiler(compiler))
            .env("RUSTC_BOOTSTRAP", "1");

        cmd.arg("-Wrustdoc::invalid_codeblock_attributes");

        if self.config.deny_warnings {
            cmd.arg("-Dwarnings");
        }
        cmd.arg("-Znormalize-docs");
        cmd.args(linker_args(self, compiler.host, LldThreads::Yes));
        cmd
    }

    /// Returns true is LLVM is enabled for the given target and we are supposed to build it.
    ///
    /// Note that this returns false if LLVM is disabled, or if we're in a
    /// check build or dry-run, where there's no need to build all of LLVM.
    pub fn is_llvm_enabled_for(&self, target: TargetSelection) -> bool {
        self.config.llvm_enabled(target) && self.kind != Kind::Check && !self.config.dry_run()
    }

    /// Return the `llvm-config` for the host target, so that it is executable.
    pub fn host_llvm_config(&self) -> PathBuf {
        self.ensure(llvm::Llvm { target: self.host_target }).llvm_config().to_owned()
    }

    /// Updates all submodules, and exits with an error if submodule
    /// management is disabled and the submodule does not exist.
    pub fn require_and_update_all_submodules(&self) {
        for submodule in self.submodule_paths() {
            self.require_submodule(submodule, None);
        }
    }

    /// Get all submodules from the src directory.
    pub fn submodule_paths(&self) -> &[String] {
        self.submodule_paths_cache.get_or_init(|| build_helper::util::parse_gitmodules(&self.src))
    }

    /// Ensure that a given step is built, returning its output. This will
    /// cache the step, so it is safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are built.
    #[track_caller]
    pub(crate) fn ensure<S: Step>(&'a self, step: S) -> S::Output {
        {
            let mut stack = self.stack.borrow_mut();
            for stack_step in stack.iter() {
                // should skip
                if stack_step.downcast_ref::<S>().is_none_or(|stack_step| *stack_step != step) {
                    continue;
                }
                let mut out = String::new();
                out += &format!("\n\nCycle in build detected when adding {step:?}\n");
                for el in stack.iter().rev() {
                    out += &format!("\t{el:?}\n");
                }
                panic!("{}", out);
            }
            if let Some(out) = self.cache.get(&step) {
                #[cfg(feature = "tracing")]
                {
                    if let Some(parent) = stack.last() {
                        let mut graph = self.sess.step_graph.borrow_mut();
                        graph.register_cached_step(&step, parent, self.config.dry_run());
                    }
                }
                return out;
            }

            #[cfg(feature = "tracing")]
            {
                let parent = stack.last();
                let mut graph = self.sess.step_graph.borrow_mut();
                graph.register_step_execution(&step, parent, self.config.dry_run());
            }

            // The location has to be gathered in this function, to be correctly propagated with
            // #[track_caller].
            let location = format_location(*std::panic::Location::caller());
            StepStack::with_current(|stack| {
                stack.push(StepRecord { info: pretty_print_step(&step), location });
            });
            stack.push(Box::new(step.clone()));
        }

        #[cfg(feature = "build-metrics")]
        self.metrics.enter_step(&step, self);

        if self.config.print_step_timings && !self.config.dry_run() {
            println!("[TIMING:start] {}", pretty_print_step(&step));
        }

        let (out, dur) = {
            let start = Instant::now();
            let zero = Duration::new(0, 0);
            let parent = self.time_spent_on_dependencies.replace(zero);

            #[cfg(feature = "tracing")]
            let _span = {
                // Keep the target and field names synchronized with `setup_tracing`.
                let span = tracing::info_span!(
                    target: STEP_SPAN_TARGET,
                    // We cannot use a dynamic name here, so instead we record the actual step name
                    // in the step_name field.
                    "step",
                    step_name = pretty_step_name::<S>(),
                    args = step_debug_args(&step),
                    location = format_location(*std::panic::Location::caller())
                );
                span.entered()
            };

            let out = step.clone().run(self);
            let dur = start.elapsed();
            let deps = self.time_spent_on_dependencies.replace(parent + dur);
            (out, dur.saturating_sub(deps))
        };

        if self.config.print_step_timings && !self.config.dry_run() {
            println!(
                "[TIMING:end] {} -- {}.{:03}",
                pretty_print_step(&step),
                dur.as_secs(),
                dur.subsec_millis()
            );
        }

        #[cfg(feature = "build-metrics")]
        self.metrics.exit_step(self);

        {
            let mut stack = self.stack.borrow_mut();
            let cur_step = stack.pop().expect("step stack empty");
            assert_eq!(cur_step.downcast_ref(), Some(&step));

            StepStack::with_current(|stack| {
                stack.pop();
            });
        }
        self.cache.put(step, out.clone());
        out
    }

    /// Ensure that a given step is built *only if it's supposed to be built by default*, returning
    /// its output. This will cache the step, so it's safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are build.
    pub(crate) fn ensure_if_default<T, S: CommandLineStep<Output = T>>(
        &'a self,
        step: S,
        kind: Kind,
    ) -> Option<S::Output> {
        let desc = CommandLineStepDescription::from::<S>(kind);
        let should_run = (desc.should_run)(ShouldRun::new(self));

        // Avoid running steps contained in --skip
        for pathset in &should_run.paths {
            if desc.is_excluded(self, pathset) {
                return None;
            }
        }

        // Only execute if it's supposed to run as default
        if (desc.is_default_step_fn)(self) { Some(self.ensure(step)) } else { None }
    }

    /// Checks if any of the "should_run" paths is in the `Builder` paths.
    pub(crate) fn was_invoked_explicitly<S: CommandLineStep>(&'a self, kind: Kind) -> bool {
        let desc = CommandLineStepDescription::from::<S>(kind);
        let should_run = (desc.should_run)(ShouldRun::new(self));

        for path in &self.paths {
            if should_run.paths.iter().any(|s| s.has(path))
                && !desc.is_excluded(self, &PathSet::Suite(TaskPath { path: path.clone() }))
            {
                return true;
            }
        }

        false
    }

    pub(crate) fn maybe_open_in_browser<S: CommandLineStep>(&self, path: impl AsRef<Path>) {
        if self.was_invoked_explicitly::<S>(Kind::Doc) {
            self.open_in_browser(path);
        } else {
            self.info(&format!("Doc path: {}", path.as_ref().display()));
        }
    }

    pub(crate) fn open_in_browser(&self, path: impl AsRef<Path>) {
        let path = path.as_ref();

        if self.config.dry_run() || !self.config.cmd.open() {
            self.info(&format!("Doc path: {}", path.display()));
            return;
        }

        self.info(&format!("Opening doc {}", path.display()));
        if let Err(err) = opener::open(path) {
            self.info(&format!("{err}\n"));
        }
    }

    pub fn exec_ctx(&self) -> &ExecutionContext {
        &self.config.exec_ctx
    }
}

/// Return qualified step name, e.g. `compile::Rustc`.
pub fn pretty_step_name<S: Step>() -> String {
    // Normalize step type path to only keep the module and the type name
    let path = type_name::<S>().rsplit("::").take(2).collect::<Vec<_>>();
    path.into_iter().rev().collect::<Vec<_>>().join("::")
}

/// Renders `step` using its `Debug` implementation and extract the field arguments out of it.
fn step_debug_args<S: Step>(step: &S) -> String {
    let step_dbg_repr = format!("{step:?}");

    // Some steps do not have any arguments, so they do not have the braces
    match (step_dbg_repr.find('{'), step_dbg_repr.rfind('}')) {
        (Some(brace_start), Some(brace_end)) => {
            step_dbg_repr[brace_start + 1..brace_end - 1].trim().to_string()
        }
        _ => String::new(),
    }
}

fn pretty_print_step<S: Step>(step: &S) -> String {
    format!("{} {{ {} }}", pretty_step_name::<S>(), step_debug_args(step))
}

impl<'a> AsRef<ExecutionContext> for Builder<'a> {
    fn as_ref(&self) -> &ExecutionContext {
        self.exec_ctx()
    }
}
