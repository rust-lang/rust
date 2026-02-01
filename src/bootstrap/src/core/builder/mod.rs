use std::any::{Any, type_name};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::fmt::{Debug, Write};
use std::hash::Hash;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::{env, fs};

use clap::ValueEnum;
#[cfg(feature = "tracing")]
use tracing::instrument;

pub use self::cargo::{Cargo, cargo_profile_var};
pub use crate::Compiler;
use crate::core::build_steps::compile::{Std, StdLink};
use crate::core::build_steps::tool::RustcPrivateCompilers;
use crate::core::build_steps::{
    check, clean, clippy, compile, dist, doc, gcc, install, llvm, run, setup, test, tool, vendor,
};
use crate::core::builder::selectors::CLIStepPath;
use crate::core::config::flags::Subcommand;
use crate::core::config::{DryRun, TargetSelection};
use crate::utils::build_stamp::BuildStamp;
use crate::utils::cache::Cache;
use crate::utils::exec::{BootstrapCommand, ExecutionContext, command};
use crate::utils::helpers::{self, LldThreads, add_dylib_path, exe, libdir, linker_args, t};
use crate::{Build, Crate, trace};

mod cargo;
mod selectors;
#[cfg(test)]
mod tests;

/// Builds and performs different [`Self::cargo_cmd`]s of stuff and actions, taking
/// into account build configuration from e.g. bootstrap.toml.
pub struct Builder<'a> {
    /// Build configuration from e.g. bootstrap.toml.
    pub build: &'a Build,

    /// The stage to use. Either implicitly determined based on subcommand, or
    /// explicitly specified with `--stage N`. Normally this is the stage we
    /// use, but sometimes we want to run steps with a lower stage than this.
    pub top_stage: u32,

    /// What to build or what action to perform.
    pub cargo_cmd: CargoSubcommand,

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

    /// Cached list of submodules from self.build.src.
    submodule_paths_cache: OnceLock<Vec<String>>,

    /// When enabled by tests, this causes the top-level steps that _would_ be
    /// executed to be logged instead. Used by snapshot tests of command-line
    /// paths-to-steps handling.
    #[expect(clippy::type_complexity)]
    log_cli_step_for_tests:
        Option<Box<dyn Fn(&StepDescription, &[StepSelectors], &[TargetSelection])>>,
}

impl Deref for Builder<'_> {
    type Target = Build;

    fn deref(&self) -> &Self::Target {
        self.build
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

pub trait Step: 'static + Clone + Debug + PartialEq + Eq + Hash {
    /// Result type of `Step::run`.
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

    /// Primary function to implement `Step` logic.
    ///
    /// This function can be triggered in two ways:
    /// 1. Directly from [`Builder::execute_cli`].
    /// 2. Indirectly by being called from other `Step`s using [`Builder::ensure`].
    ///
    /// When called with [`Builder::execute_cli`] (as done by `Build::build`), this function is executed twice:
    /// - First in "dry-run" mode to validate certain things (like cyclic Step invocations,
    ///   directory creation, etc) super quickly.
    /// - Then it's called again to run the actual, very expensive process.
    ///
    /// When triggered indirectly from other `Step`s, it may still run twice (as dry-run and real mode)
    /// depending on the `Step::run` implementation of the caller.
    fn run(self, builder: &Builder<'_>) -> Self::Output;

    /// Called directly by the bootstrap `Step` handler when not triggered indirectly by other `Step`s using [`Builder::ensure`].
    /// For example, `./x.py test bootstrap` runs this for `test::Bootstrap`. Similarly, `./x.py test` runs it for every step
    /// that is listed by the `describe` macro in [`Builder::get_step_descriptions`].
    fn make_run(_run: RunConfig<'_>) {
        // It is reasonable to not have an implementation of make_run for rules
        // who do not want to get called from the root context. This means that
        // they are likely dependencies (e.g., sysroot creation) or similar, and
        // as such calling them from ./x.py isn't logical.
        unimplemented!()
    }

    /// Returns metadata of the step, for tests
    fn metadata(&self) -> Option<StepMetadata> {
        None
    }
}

/// Metadata that describes an executed step, mostly for testing and tracing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepMetadata {
    name: String,
    cargo_cmd: CargoSubcommand,
    target: TargetSelection,
    built_by: Option<Compiler>,
    stage: Option<u32>,
    /// Additional opaque string printed in the metadata
    metadata: Option<String>,
}

impl StepMetadata {
    pub fn build(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Build)
    }

    pub fn check(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Check)
    }

    pub fn clippy(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Clippy)
    }

    pub fn doc(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Doc)
    }

    pub fn dist(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Dist)
    }

    pub fn test(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Test)
    }

    pub fn run(name: &str, target: TargetSelection) -> Self {
        Self::new(name, target, CargoSubcommand::Run)
    }

    fn new(name: &str, target: TargetSelection, cargo_cmd: CargoSubcommand) -> Self {
        Self {
            name: name.to_string(),
            cargo_cmd,
            target,
            built_by: None,
            stage: None,
            metadata: None,
        }
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

    pub fn get_stage(&self) -> Option<u32> {
        self.stage.or(self
            .built_by
            // For std, its stage corresponds to the stage of the compiler that builds it.
            // For everything else, a stage N things gets built by a stage N-1 compiler.
            .map(|compiler| if self.name == "std" { compiler.stage } else { compiler.stage + 1 }))
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_target(&self) -> TargetSelection {
        self.target
    }
}

pub struct RunConfig<'a> {
    pub builder: &'a Builder<'a>,
    pub target: TargetSelection,
    pub paths: Vec<StepSelectors>,
}

impl RunConfig<'_> {
    pub fn build_triple(&self) -> TargetSelection {
        self.builder.build.host_target
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
    pub fn expand_alias(&self, alias: Alias) -> Vec<String> {
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

struct StepDescription {
    is_host: bool,
    should_run: fn(ShouldRun<'_>) -> ShouldRun<'_>,
    is_default_step_fn: fn(&Builder<'_>) -> bool,
    make_run: fn(RunConfig<'_>),
    name: &'static str,
    cargo_cmd: CargoSubcommand,
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct StepSelection {
    pub path: PathBuf,
    pub cargo_cmd: Option<CargoSubcommand>,
}

impl Debug for StepSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(cmd) = &self.cargo_cmd {
            write!(f, "{}::", cmd.as_str())?;
        }
        write!(f, "{}", self.path.display())
    }
}

/// Collection of paths used to match a task rule.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum StepSelectors {
    /// A collection of individual paths or aliases.
    ///
    /// These are generally matched as a path suffix. For example, a
    /// command-line value of `std` will match if `library/std` is in the
    /// set.
    ///
    /// NOTE: the paths within a set should always be aliases of one another.
    /// For example, `src/librustdoc` and `src/tools/rustdoc` should be in the same set,
    /// but `library/core` and `library/std` generally should not, unless there's no way (for that Step)
    /// to build them separately.
    Alias(BTreeSet<StepSelection>),
    /// A "suite" of paths.
    ///
    /// These can match as a path suffix (like `Set`), or as a prefix. For
    /// example, a command-line value of `tests/ui/abi/variadic-ffi.rs`
    /// will match `tests/ui`. A command-line value of `ui` would also
    /// match `tests/ui`.
    TestSuite(StepSelection),
}

impl StepSelectors {
    fn empty() -> StepSelectors {
        StepSelectors::Alias(BTreeSet::new())
    }

    fn one<P: Into<PathBuf>>(path: P, cargo_cmd: CargoSubcommand) -> StepSelectors {
        let mut set = BTreeSet::new();
        set.insert(StepSelection { path: path.into(), cargo_cmd: Some(cargo_cmd) });
        StepSelectors::Alias(set)
    }

    fn has(&self, needle: &Path, module: CargoSubcommand) -> bool {
        match self {
            StepSelectors::Alias(set) => set.iter().any(|p| Self::check(p, needle, module)),
            StepSelectors::TestSuite(suite) => Self::check(suite, needle, module),
        }
    }

    // internal use only
    fn check(p: &StepSelection, needle: &Path, module: CargoSubcommand) -> bool {
        let check_path = || {
            // This order is important for retro-compatibility, as `starts_with` was introduced later.
            p.path.ends_with(needle) || p.path.starts_with(needle)
        };
        if let Some(p_kind) = &p.cargo_cmd { check_path() && *p_kind == module } else { check_path() }
    }

    /// Return all `TaskPath`s in `Self` that contain any of the `needles`, removing the
    /// matched needles.
    ///
    /// This is used for `StepDescription::krate`, which passes all matching crates at once to
    /// `Step::make_run`, rather than calling it many times with a single crate.
    /// See `tests.rs` for examples.
    fn intersection_removing_matches(&self, needles: &mut [CLIStepPath], module: CargoSubcommand) -> StepSelectors {
        let mut check = |p| {
            let mut result = false;
            for n in needles.iter_mut() {
                let matched = Self::check(p, &n.path, module);
                if matched {
                    n.will_be_executed = true;
                    result = true;
                }
            }
            result
        };
        match self {
            StepSelectors::Alias(set) => StepSelectors::Alias(set.iter().filter(|&p| check(p)).cloned().collect()),
            StepSelectors::TestSuite(suite) => {
                if check(suite) {
                    self.clone()
                } else {
                    StepSelectors::empty()
                }
            }
        }
    }

    /// A convenience wrapper for Steps which know they have no aliases and all their sets contain only a single path.
    ///
    /// This can be used with [`ShouldRun::crate_or_deps`], [`ShouldRun::path`], or [`ShouldRun::alias`].
    #[track_caller]
    pub fn assert_single_path(&self) -> &StepSelection {
        match self {
            StepSelectors::Alias(set) => {
                assert_eq!(set.len(), 1, "called assert_single_path on multiple paths");
                set.iter().next().unwrap()
            }
            StepSelectors::TestSuite(_) => unreachable!("called assert_single_path on a Suite path"),
        }
    }
}

impl StepDescription {
    fn from<S: Step>(cargo_cmd: CargoSubcommand) -> StepDescription {
        StepDescription {
            is_host: S::IS_HOST,
            should_run: S::should_run,
            is_default_step_fn: S::is_default_step,
            make_run: S::make_run,
            name: std::any::type_name::<S>(),
            cargo_cmd,
        }
    }

    fn maybe_run(&self, builder: &Builder<'_>, mut pathsets: Vec<StepSelectors>) {
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

    fn is_excluded(&self, builder: &Builder<'_>, pathset: &StepSelectors) -> bool {
        if builder.config.skip.iter().any(|e| pathset.has(e, builder.cargo_cmd)) {
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
    cargo_cmd: CargoSubcommand,

    // use a BTreeSet to maintain sort order
    paths: BTreeSet<StepSelectors>,
}

impl<'a> ShouldRun<'a> {
    fn new(builder: &'a Builder<'_>, cargo_cmd: CargoSubcommand) -> ShouldRun<'a> {
        ShouldRun { builder, cargo_cmd, paths: BTreeSet::new() }
    }

    /// Indicates it should run if the command-line selects the given crate or
    /// any of its (local) dependencies.
    ///
    /// `make_run` will be called a single time with all matching command-line paths.
    pub fn crate_or_deps(self, name: &str) -> Self {
        let crates = self.builder.in_tree_crates(name, None);
        self.crates(crates)
    }

    /// Indicates it should run if the command-line selects any of the given crates.
    ///
    /// `make_run` will be called a single time with all matching command-line paths.
    ///
    /// Prefer [`ShouldRun::crate_or_deps`] to this function where possible.
    pub(crate) fn crates(mut self, crates: Vec<&Crate>) -> Self {
        for krate in crates {
            let path = krate.local_path(self.builder);
            self.paths.insert(StepSelectors::one(path, self.cargo_cmd));
        }
        self
    }

    // single alias, which does not correspond to any on-disk path
    pub fn alias(mut self, alias: &str) -> Self {
        // exceptional case for `Kind::Setup` because its `library`
        // and `compiler` options would otherwise naively match with
        // `compiler` and `library` folders respectively.
        assert!(
            self.cargo_cmd == CargoSubcommand::Setup || !self.builder.src.join(alias).exists(),
            "use `builder.path()` for real paths: {alias}"
        );
        self.paths.insert(StepSelectors::Alias(
            std::iter::once(StepSelection { path: alias.into(), cargo_cmd: Some(self.cargo_cmd) }).collect(),
        ));
        self
    }

    /// single, non-aliased path
    ///
    /// Must be an on-disk path; use `alias` for names that do not correspond to on-disk paths.
    pub fn path(mut self, path: &str) -> Self {
        let submodules_paths = self.builder.submodule_paths();

        // assert only if `p` isn't submodule
        if !submodules_paths.iter().any(|sm_p| path.contains(sm_p)) {
            assert!(
                self.builder.src.join(path).exists(),
                "`should_run.paths` should correspond to real on-disk paths - use `alias` if there is no relevant path: {path}"
            );
        }

        let task = StepSelection { path: path.into(), cargo_cmd: Some(self.cargo_cmd) };
        self.paths.insert(StepSelectors::Alias(BTreeSet::from_iter([task])));
        self
    }

    /// Handles individual files (not directories) within a test suite.
    fn is_suite_path(&self, requested_path: &Path) -> Option<&StepSelectors> {
        self.paths.iter().find(|pathset| match pathset {
            StepSelectors::TestSuite(suite) => requested_path.starts_with(&suite.path),
            StepSelectors::Alias(_) => false,
        })
    }

    pub fn suite_path(mut self, suite: &str) -> Self {
        self.paths.insert(StepSelectors::TestSuite(StepSelection { path: suite.into(), cargo_cmd: Some(self.cargo_cmd) }));
        self
    }

    // allows being more explicit about why should_run in Step returns the value passed to it
    pub fn never(mut self) -> ShouldRun<'a> {
        self.paths.insert(StepSelectors::empty());
        self
    }

    /// Given a set of requested paths, return the subset which match the Step for this `ShouldRun`,
    /// removing the matches from `paths`.
    ///
    /// NOTE: this returns multiple PathSets to allow for the possibility of multiple units of work
    /// within the same step. For example, `test::Crate` allows testing multiple crates in the same
    /// cargo invocation, which are put into separate sets because they aren't aliases.
    ///
    /// The reason we return PathSet instead of PathBuf is to allow for aliases that mean the same thing
    /// (for now, just `all_krates` and `paths`, but we may want to add an `aliases` function in the future?)
    fn pathset_for_paths_removing_matches(
        &self,
        paths: &mut [CLIStepPath],
        cargo_cmd: CargoSubcommand,
    ) -> Vec<StepSelectors> {
        let mut sets = vec![];
        for pathset in &self.paths {
            let subset = pathset.intersection_removing_matches(paths, cargo_cmd);
            if subset != StepSelectors::empty() {
                sets.push(subset);
            }
        }
        sets
    }
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, PartialOrd, Ord, ValueEnum)]
pub enum CargoSubcommand {
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

impl CargoSubcommand {
    pub fn as_str(&self) -> &'static str {
        match self {
            CargoSubcommand::Build => "build",
            CargoSubcommand::Check => "check",
            CargoSubcommand::Clippy => "clippy",
            CargoSubcommand::Fix => "fix",
            CargoSubcommand::Format => "fmt",
            CargoSubcommand::Test => "test",
            CargoSubcommand::Miri => "miri",
            CargoSubcommand::MiriSetup => panic!("`as_str` is not supported for `MiriSetup`."),
            CargoSubcommand::MiriTest => panic!("`as_str` is not supported for `MiriTest`."),
            CargoSubcommand::Bench => "bench",
            CargoSubcommand::Doc => "doc",
            CargoSubcommand::Clean => "clean",
            CargoSubcommand::Dist => "dist",
            CargoSubcommand::Install => "install",
            CargoSubcommand::Run => "run",
            CargoSubcommand::Setup => "setup",
            CargoSubcommand::Vendor => "vendor",
            CargoSubcommand::Perf => "perf",
        }
    }

    pub fn description(&self) -> String {
        match self {
            CargoSubcommand::Test => "Testing",
            CargoSubcommand::Bench => "Benchmarking",
            CargoSubcommand::Doc => "Documenting",
            CargoSubcommand::Run => "Running",
            CargoSubcommand::Clippy => "Linting",
            CargoSubcommand::Perf => "Profiling & benchmarking",
            _ => {
                let title_letter = self.as_str()[0..1].to_ascii_uppercase();
                return format!("{title_letter}{}ing", &self.as_str()[1..]);
            }
        }
        .to_owned()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Libdir {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for Libdir {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let relative_sysroot_libdir = builder.sysroot_libdir_relative(self.compiler);
        let sysroot = builder.sysroot(self.compiler).join(relative_sysroot_libdir).join("rustlib");

        if !builder.config.dry_run() {
            // Avoid deleting the `rustlib/` directory we just copied (in `impl Step for
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
    fn get_step_descriptions(cargo_cmd: CargoSubcommand) -> Vec<StepDescription> {
        macro_rules! describe {
            ($($rule:ty),+ $(,)?) => {{
                vec![$(StepDescription::from::<$rule>(cargo_cmd)),+]
            }};
        }
        match cargo_cmd {
            CargoSubcommand::Build => describe!(
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
                llvm::CrtBeginEnd,
                tool::RustdocGUITest,
                tool::OptimizedDist,
                tool::CoverageDump,
                tool::LlvmBitcodeLinker,
                tool::RustcPerf,
                tool::WasmComponentLd,
                tool::LldWrapper
            ),
            CargoSubcommand::Clippy => describe!(
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
            CargoSubcommand::Check | CargoSubcommand::Fix => describe!(
                check::Rustc,
                check::Rustdoc,
                check::CraneliftCodegenBackend,
                check::GccCodegenBackend,
                check::Clippy,
                check::Miri,
                check::CargoMiri,
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
            CargoSubcommand::Test => describe!(
                crate::core::build_steps::toolstate::ToolStateCheck,
                test::Tidy,
                test::BootstrapPy,
                test::Bootstrap,
                test::Ui,
                test::Crashes,
                test::Coverage,
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
                test::Clippy,
                test::CompiletestTest,
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
            ),
            CargoSubcommand::Miri => describe!(test::Crate),
            CargoSubcommand::Bench => {
                describe!(test::Crate, test::CrateLibrustc, test::CrateRustdoc)
            }
            CargoSubcommand::Doc => describe!(
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
            CargoSubcommand::Dist => describe!(
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
            CargoSubcommand::Install => describe!(
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
            CargoSubcommand::Run => describe!(
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
            CargoSubcommand::Setup => {
                describe!(setup::Profile, setup::Hook, setup::Link, setup::Editor)
            }
            CargoSubcommand::Clean => describe!(clean::CleanAll, clean::Rustc, clean::Std),
            CargoSubcommand::Vendor => describe!(vendor::Vendor),
            // special-cased in Build::build()
            CargoSubcommand::Format | CargoSubcommand::Perf => vec![],
            CargoSubcommand::MiriTest | CargoSubcommand::MiriSetup => unreachable!(),
        }
    }

    pub fn get_help(build: &Build, cargo_cmd: CargoSubcommand) -> Option<String> {
        let step_descriptions = Builder::get_step_descriptions(cargo_cmd);
        if step_descriptions.is_empty() {
            return None;
        }

        let builder = Self::new_internal(build, cargo_cmd, vec![]);
        let builder = &builder;
        // The "build" cmd here is just a placeholder, it will be replaced with something else in
        // the following statement.
        let mut should_run = ShouldRun::new(builder, CargoSubcommand::Build);
        for desc in step_descriptions {
            should_run.cargo_cmd = desc.cargo_cmd;
            should_run = (desc.should_run)(should_run);
        }
        let mut help = String::from("Available paths:\n");
        let mut add_path = |path: &Path| {
            t!(write!(help, "    ./x.py {} {}\n", cargo_cmd.as_str(), path.display()));
        };
        for pathset in should_run.paths {
            match pathset {
                StepSelectors::Alias(set) => {
                    for path in set {
                        add_path(&path.path);
                    }
                }
                StepSelectors::TestSuite(path) => {
                    add_path(&path.path.join("..."));
                }
            }
        }
        Some(help)
    }

    fn new_internal(build: &Build, cargo_cmd: CargoSubcommand, paths: Vec<PathBuf>) -> Builder<'_> {
        Builder {
            build,
            top_stage: build.config.stage,
            cargo_cmd,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
            time_spent_on_dependencies: Cell::new(Duration::new(0, 0)),
            paths,
            submodule_paths_cache: Default::default(),
            log_cli_step_for_tests: None,
        }
    }

    pub fn new(build: &Build) -> Builder<'_> {
        let paths = &build.config.paths;
        let (cargo_cmd, paths) = match build.config.cmd {
            Subcommand::Build { .. } => (CargoSubcommand::Build, &paths[..]),
            Subcommand::Check { .. } => (CargoSubcommand::Check, &paths[..]),
            Subcommand::Clippy { .. } => (CargoSubcommand::Clippy, &paths[..]),
            Subcommand::Fix => (CargoSubcommand::Fix, &paths[..]),
            Subcommand::Doc { .. } => (CargoSubcommand::Doc, &paths[..]),
            Subcommand::Test { .. } => (CargoSubcommand::Test, &paths[..]),
            Subcommand::Miri { .. } => (CargoSubcommand::Miri, &paths[..]),
            Subcommand::Bench { .. } => (CargoSubcommand::Bench, &paths[..]),
            Subcommand::Dist => (CargoSubcommand::Dist, &paths[..]),
            Subcommand::Install => (CargoSubcommand::Install, &paths[..]),
            Subcommand::Run { .. } => (CargoSubcommand::Run, &paths[..]),
            Subcommand::Clean { .. } => (CargoSubcommand::Clean, &paths[..]),
            Subcommand::Format { .. } => (CargoSubcommand::Format, &[][..]),
            Subcommand::Setup { profile: ref path } => (
                CargoSubcommand::Setup,
                path.as_ref().map_or([].as_slice(), |path| std::slice::from_ref(path)),
            ),
            Subcommand::Vendor { .. } => (CargoSubcommand::Vendor, &paths[..]),
            Subcommand::Perf { .. } => (CargoSubcommand::Perf, &paths[..]),
        };

        Self::new_internal(build, cargo_cmd, paths.to_owned())
    }

    pub fn execute_cli(&self) {
        self.run_step_descriptions(&Builder::get_step_descriptions(self.cargo_cmd), &self.paths);
    }

    /// Run all default documentation steps to build documentation.
    pub fn run_default_doc_steps(&self) {
        self.run_step_descriptions(&Builder::get_step_descriptions(CargoSubcommand::Doc), &[]);
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

    fn run_step_descriptions(&self, v: &[StepDescription], paths: &[PathBuf]) {
        selectors::match_paths_to_steps_and_run(self, v, paths);
    }

    /// Returns if `std` should be statically linked into `rustc_driver`.
    /// It's currently not done on `windows-gnu` due to linker bugs.
    pub fn link_std_into_rustc_driver(&self, target: TargetSelection) -> bool {
        !target.triple.ends_with("-windows-gnu")
    }

    /// Obtain a compiler at a given stage and for a given host (i.e., this is the target that the
    /// compiler will run on, *not* the target it will build code for). Explicitly does not take
    /// `Compiler` since all `Compiler` instances are meant to be obtained through this function,
    /// since it ensures that they are valid (i.e., built and assembled).
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
        let mut resolved_compiler = if self.build.force_use_stage2(stage) {
            trace!(target: "COMPILER_FOR", ?stage, "force_use_stage2");
            self.compiler(2, self.config.host_target)
        } else if self.build.force_use_stage1(stage, target) {
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

    pub fn sysroot(&self, compiler: Compiler) -> PathBuf {
        self.ensure(compile::Sysroot::new(compiler))
    }

    /// Returns the bindir for a compiler's sysroot.
    pub fn sysroot_target_bindir(&self, compiler: Compiler, target: TargetSelection) -> PathBuf {
        self.ensure(Libdir { compiler, target }).join(target).join("bin")
    }

    /// Returns the libdir where the standard library and other artifacts are
    /// found for a compiler's sysroot.
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
            _ if compiler.stage == 0 => &self.build.initial_relative_libdir,
            _ => Path::new("lib"),
        }
    }

    pub fn rustc_lib_paths(&self, compiler: Compiler) -> Vec<PathBuf> {
        let mut dylib_dirs = vec![self.rustc_libdir(compiler)];

        // Ensure that the downloaded LLVM libraries can be found.
        if self.config.llvm_from_ci {
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
            .map(|entry| entry.path())
    }

    /// Returns a path to `Rustdoc` that "belongs" to the `target_compiler`.
    /// It can be either a stage0 rustdoc or a locally built rustdoc that *links* to
    /// `target_compiler`.
    pub fn rustdoc_for_compiler(&self, target_compiler: Compiler) -> PathBuf {
        self.ensure(tool::Rustdoc { target_compiler })
    }

    pub fn cargo_miri_cmd(&self, run_compiler: Compiler) -> BootstrapCommand {
        assert!(run_compiler.stage > 0, "miri can not be invoked at stage 0");

        let compilers =
            RustcPrivateCompilers::new(self, run_compiler.stage, self.build.host_target);
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
            let cargo_clippy = self
                .config
                .initial_cargo_clippy
                .clone()
                .unwrap_or_else(|| self.build.config.download_clippy());

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

    /// Return the path to `llvm-config` for the target, if it exists.
    ///
    /// Note that this returns `None` if LLVM is disabled, or if we're in a
    /// check build or dry-run, where there's no need to build all of LLVM.
    ///
    /// FIXME(@kobzol)
    /// **WARNING**: This actually returns the **HOST** LLVM config, not LLVM config for the given
    /// *target*.
    pub fn llvm_config(&self, target: TargetSelection) -> Option<PathBuf> {
        if self.config.llvm_enabled(target)
            && self.cargo_cmd != CargoSubcommand::Check
            && !self.config.dry_run()
        {
            let llvm::LlvmResult { host_llvm_config, .. } = self.ensure(llvm::Llvm { target });
            if host_llvm_config.is_file() {
                return Some(host_llvm_config);
            }
        }
        None
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
    pub fn ensure<S: Step>(&'a self, step: S) -> S::Output {
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
                        let mut graph = self.build.step_graph.borrow_mut();
                        graph.register_cached_step(&step, parent, self.config.dry_run());
                    }
                }
                return out;
            }

            #[cfg(feature = "tracing")]
            {
                let parent = stack.last();
                let mut graph = self.build.step_graph.borrow_mut();
                graph.register_step_execution(&step, parent, self.config.dry_run());
            }

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
                    args = step_debug_args(&step)
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
        }
        self.cache.put(step, out.clone());
        out
    }

    /// Ensure that a given step is built *only if it's supposed to be built by default*, returning
    /// its output. This will cache the step, so it's safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are build.
    pub(crate) fn ensure_if_default<T, S: Step<Output = T>>(
        &'a self,
        step: S,
        cargo_cmd: CargoSubcommand,
    ) -> Option<S::Output> {
        let desc = StepDescription::from::<S>(cargo_cmd);
        let should_run = (desc.should_run)(ShouldRun::new(self, desc.cargo_cmd));

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
    pub(crate) fn was_invoked_explicitly<S: Step>(&'a self, cargo_cmd: CargoSubcommand) -> bool {
        let desc = StepDescription::from::<S>(cargo_cmd);
        let should_run = (desc.should_run)(ShouldRun::new(self, desc.cargo_cmd));

        for path in &self.paths {
            if should_run.paths.iter().any(|s| s.has(path, desc.cargo_cmd))
                && !desc.is_excluded(
                    self,
                    &StepSelectors::TestSuite(StepSelection { path: path.clone(), cargo_cmd: Some(desc.cargo_cmd) }),
                )
            {
                return true;
            }
        }

        false
    }

    pub(crate) fn maybe_open_in_browser<S: Step>(&self, path: impl AsRef<Path>) {
        if self.was_invoked_explicitly::<S>(CargoSubcommand::Doc) {
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
