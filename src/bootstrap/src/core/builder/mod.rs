use std::any::{Any, type_name};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::fmt::{self, Debug, Write};
use std::hash::Hash;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant};
use std::{env, fs};

use clap::ValueEnum;
#[cfg(feature = "tracing")]
use tracing::instrument;

pub use self::cargo::{Cargo, cargo_profile_var};
pub use crate::Compiler;
use crate::core::build_steps::{
    check, clean, clippy, compile, dist, doc, gcc, install, llvm, run, setup, test, tool, vendor,
};
use crate::core::config::flags::Subcommand;
use crate::core::config::{DryRun, TargetSelection};
use crate::utils::cache::Cache;
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::execution_context::ExecutionContext;
use crate::utils::helpers::{self, LldThreads, add_dylib_path, exe, libdir, linker_args, t};
use crate::{Build, Crate, trace};

mod cargo;

#[cfg(test)]
mod tests;

/// Builds and performs different [`Self::kind`]s of stuff and actions, taking
/// into account build configuration from e.g. bootstrap.toml.
pub struct Builder<'a> {
    /// Build configuration from e.g. bootstrap.toml.
    pub build: &'a Build,

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

    /// Cached list of submodules from self.build.src.
    submodule_paths_cache: OnceLock<Vec<String>>,
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
trait AnyDebug: Any + Debug {}
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

    /// Whether this step is run by default as part of its respective phase, as defined by the `describe`
    /// macro in [`Builder::get_step_descriptions`].
    ///
    /// Note: Even if set to `true`, it can still be overridden with [`ShouldRun::default_condition`]
    /// by `Step::should_run`.
    const DEFAULT: bool = false;

    /// If true, then this rule should be skipped if --target was specified, but --host was not
    const ONLY_HOSTS: bool = false;

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

    /// Determines if this `Step` should be run when given specific paths (e.g., `x build $path`).
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_>;

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
#[allow(unused)]
#[derive(Debug)]
pub struct StepMetadata {
    name: &'static str,
    kind: Kind,
    target: TargetSelection,
    built_by: Option<Compiler>,
    stage: Option<u32>,
}

impl StepMetadata {
    pub fn build(name: &'static str, target: TargetSelection) -> Self {
        Self { name, kind: Kind::Build, target, built_by: None, stage: None }
    }

    pub fn built_by(mut self, compiler: Compiler) -> Self {
        self.built_by = Some(compiler);
        self
    }

    pub fn stage(mut self, stage: u32) -> Self {
        self.stage = Some(stage);
        self
    }
}

pub struct RunConfig<'a> {
    pub builder: &'a Builder<'a>,
    pub target: TargetSelection,
    pub paths: Vec<PathSet>,
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

    let mut descr = String::from(" {");
    descr.push_str(crates[0].as_ref());
    for krate in &crates[1..] {
        descr.push_str(", ");
        descr.push_str(krate.as_ref());
    }
    descr.push('}');
    descr
}

struct StepDescription {
    default: bool,
    only_hosts: bool,
    should_run: fn(ShouldRun<'_>) -> ShouldRun<'_>,
    make_run: fn(RunConfig<'_>),
    name: &'static str,
    kind: Kind,
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct TaskPath {
    pub path: PathBuf,
    pub kind: Option<Kind>,
}

impl Debug for TaskPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(kind) = &self.kind {
            write!(f, "{}::", kind.as_str())?;
        }
        write!(f, "{}", self.path.display())
    }
}

/// Collection of paths used to match a task rule.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum PathSet {
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
    fn empty() -> PathSet {
        PathSet::Set(BTreeSet::new())
    }

    fn one<P: Into<PathBuf>>(path: P, kind: Kind) -> PathSet {
        let mut set = BTreeSet::new();
        set.insert(TaskPath { path: path.into(), kind: Some(kind) });
        PathSet::Set(set)
    }

    fn has(&self, needle: &Path, module: Kind) -> bool {
        match self {
            PathSet::Set(set) => set.iter().any(|p| Self::check(p, needle, module)),
            PathSet::Suite(suite) => Self::check(suite, needle, module),
        }
    }

    // internal use only
    fn check(p: &TaskPath, needle: &Path, module: Kind) -> bool {
        let check_path = || {
            // This order is important for retro-compatibility, as `starts_with` was introduced later.
            p.path.ends_with(needle) || p.path.starts_with(needle)
        };
        if let Some(p_kind) = &p.kind { check_path() && *p_kind == module } else { check_path() }
    }

    /// Return all `TaskPath`s in `Self` that contain any of the `needles`, removing the
    /// matched needles.
    ///
    /// This is used for `StepDescription::krate`, which passes all matching crates at once to
    /// `Step::make_run`, rather than calling it many times with a single crate.
    /// See `tests.rs` for examples.
    fn intersection_removing_matches(&self, needles: &mut [CLIStepPath], module: Kind) -> PathSet {
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
            PathSet::Set(set) => PathSet::Set(set.iter().filter(|&p| check(p)).cloned().collect()),
            PathSet::Suite(suite) => {
                if check(suite) {
                    self.clone()
                } else {
                    PathSet::empty()
                }
            }
        }
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

const PATH_REMAP: &[(&str, &[&str])] = &[
    // bootstrap.toml uses `rust-analyzer-proc-macro-srv`, but the
    // actual path is `proc-macro-srv-cli`
    ("rust-analyzer-proc-macro-srv", &["src/tools/rust-analyzer/crates/proc-macro-srv-cli"]),
    // Make `x test tests` function the same as `x t tests/*`
    (
        "tests",
        &[
            // tidy-alphabetical-start
            "tests/assembly",
            "tests/codegen",
            "tests/codegen-units",
            "tests/coverage",
            "tests/coverage-run-rustdoc",
            "tests/crashes",
            "tests/debuginfo",
            "tests/incremental",
            "tests/mir-opt",
            "tests/pretty",
            "tests/run-make",
            "tests/rustdoc",
            "tests/rustdoc-gui",
            "tests/rustdoc-js",
            "tests/rustdoc-js-std",
            "tests/rustdoc-json",
            "tests/rustdoc-ui",
            "tests/ui",
            "tests/ui-fulldeps",
            // tidy-alphabetical-end
        ],
    ),
];

fn remap_paths(paths: &mut Vec<PathBuf>) {
    let mut remove = vec![];
    let mut add = vec![];
    for (i, path) in paths.iter().enumerate().filter_map(|(i, path)| path.to_str().map(|s| (i, s)))
    {
        for &(search, replace) in PATH_REMAP {
            // Remove leading and trailing slashes so `tests/` and `tests` are equivalent
            if path.trim_matches(std::path::is_separator) == search {
                remove.push(i);
                add.extend(replace.iter().map(PathBuf::from));
                break;
            }
        }
    }
    remove.sort();
    remove.dedup();
    for idx in remove.into_iter().rev() {
        paths.remove(idx);
    }
    paths.append(&mut add);
}

#[derive(Clone, PartialEq)]
struct CLIStepPath {
    path: PathBuf,
    will_be_executed: bool,
}

#[cfg(test)]
impl CLIStepPath {
    fn will_be_executed(mut self, will_be_executed: bool) -> Self {
        self.will_be_executed = will_be_executed;
        self
    }
}

impl Debug for CLIStepPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.display())
    }
}

impl From<PathBuf> for CLIStepPath {
    fn from(path: PathBuf) -> Self {
        Self { path, will_be_executed: false }
    }
}

impl StepDescription {
    fn from<S: Step>(kind: Kind) -> StepDescription {
        StepDescription {
            default: S::DEFAULT,
            only_hosts: S::ONLY_HOSTS,
            should_run: S::should_run,
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
        let targets = if self.only_hosts { &builder.hosts } else { &builder.targets };

        for target in targets {
            let run = RunConfig { builder, paths: pathsets.clone(), target: *target };
            (self.make_run)(run);
        }
    }

    fn is_excluded(&self, builder: &Builder<'_>, pathset: &PathSet) -> bool {
        if builder.config.skip.iter().any(|e| pathset.has(e, builder.kind)) {
            if !matches!(builder.config.get_dry_run(), DryRun::SelfCheck) {
                println!("Skipping {pathset:?} because it is excluded");
            }
            return true;
        }

        if !builder.config.skip.is_empty()
            && !matches!(builder.config.get_dry_run(), DryRun::SelfCheck)
        {
            builder.verbose(|| {
                println!(
                    "{:?} not skipped for {:?} -- not in {:?}",
                    pathset, self.name, builder.config.skip
                )
            });
        }
        false
    }

    fn run(v: &[StepDescription], builder: &Builder<'_>, paths: &[PathBuf]) {
        let should_runs = v
            .iter()
            .map(|desc| (desc.should_run)(ShouldRun::new(builder, desc.kind)))
            .collect::<Vec<_>>();

        if builder.download_rustc() && (builder.kind == Kind::Dist || builder.kind == Kind::Install)
        {
            eprintln!(
                "ERROR: '{}' subcommand is incompatible with `rust.download-rustc`.",
                builder.kind.as_str()
            );
            crate::exit!(1);
        }

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
                    desc.maybe_run(builder, should_run.paths.iter().cloned().collect());
                }
            }
        }

        // Attempt to resolve paths to be relative to the builder source directory.
        let mut paths: Vec<PathBuf> = paths
            .iter()
            .map(|p| {
                // If the path does not exist, it may represent the name of a Step, such as `tidy` in `x test tidy`
                if !p.exists() {
                    return p.clone();
                }

                // Make the path absolute, strip the prefix, and convert to a PathBuf.
                match std::path::absolute(p) {
                    Ok(p) => p.strip_prefix(&builder.src).unwrap_or(&p).to_path_buf(),
                    Err(e) => {
                        eprintln!("ERROR: {e:?}");
                        panic!("Due to the above error, failed to resolve path: {p:?}");
                    }
                }
            })
            .collect();

        remap_paths(&mut paths);

        // Handle all test suite paths.
        // (This is separate from the loop below to avoid having to handle multiple paths in `is_suite_path` somehow.)
        paths.retain(|path| {
            for (desc, should_run) in v.iter().zip(&should_runs) {
                if let Some(suite) = should_run.is_suite_path(path) {
                    desc.maybe_run(builder, vec![suite.clone()]);
                    return false;
                }
            }
            true
        });

        if paths.is_empty() {
            return;
        }

        let mut paths: Vec<CLIStepPath> = paths.into_iter().map(|p| p.into()).collect();
        let mut path_lookup: Vec<(CLIStepPath, bool)> =
            paths.clone().into_iter().map(|p| (p, false)).collect();

        // List of `(usize, &StepDescription, Vec<PathSet>)` where `usize` is the closest index of a path
        // compared to the given CLI paths. So we can respect to the CLI order by using this value to sort
        // the steps.
        let mut steps_to_run = vec![];

        for (desc, should_run) in v.iter().zip(&should_runs) {
            let pathsets = should_run.pathset_for_paths_removing_matches(&mut paths, desc.kind);

            // This value is used for sorting the step execution order.
            // By default, `usize::MAX` is used as the index for steps to assign them the lowest priority.
            //
            // If we resolve the step's path from the given CLI input, this value will be updated with
            // the step's actual index.
            let mut closest_index = usize::MAX;

            // Find the closest index from the original list of paths given by the CLI input.
            for (index, (path, is_used)) in path_lookup.iter_mut().enumerate() {
                if !*is_used && !paths.contains(path) {
                    closest_index = index;
                    *is_used = true;
                    break;
                }
            }

            steps_to_run.push((closest_index, desc, pathsets));
        }

        // Sort the steps before running them to respect the CLI order.
        steps_to_run.sort_by_key(|(index, _, _)| *index);

        // Handle all PathSets.
        for (_index, desc, pathsets) in steps_to_run {
            if !pathsets.is_empty() {
                desc.maybe_run(builder, pathsets);
            }
        }

        paths.retain(|p| !p.will_be_executed);

        if !paths.is_empty() {
            eprintln!("ERROR: no `{}` rules matched {:?}", builder.kind.as_str(), paths);
            eprintln!(
                "HELP: run `x.py {} --help --verbose` to show a list of available paths",
                builder.kind.as_str()
            );
            eprintln!(
                "NOTE: if you are adding a new Step to bootstrap itself, make sure you register it with `describe!`"
            );
            crate::exit!(1);
        }
    }
}

enum ReallyDefault<'a> {
    Bool(bool),
    Lazy(LazyLock<bool, Box<dyn Fn() -> bool + 'a>>),
}

pub struct ShouldRun<'a> {
    pub builder: &'a Builder<'a>,
    kind: Kind,

    // use a BTreeSet to maintain sort order
    paths: BTreeSet<PathSet>,

    // If this is a default rule, this is an additional constraint placed on
    // its run. Generally something like compiler docs being enabled.
    is_really_default: ReallyDefault<'a>,
}

impl<'a> ShouldRun<'a> {
    fn new(builder: &'a Builder<'_>, kind: Kind) -> ShouldRun<'a> {
        ShouldRun {
            builder,
            kind,
            paths: BTreeSet::new(),
            is_really_default: ReallyDefault::Bool(true), // by default no additional conditions
        }
    }

    pub fn default_condition(mut self, cond: bool) -> Self {
        self.is_really_default = ReallyDefault::Bool(cond);
        self
    }

    pub fn lazy_default_condition(mut self, lazy_cond: Box<dyn Fn() -> bool + 'a>) -> Self {
        self.is_really_default = ReallyDefault::Lazy(LazyLock::new(lazy_cond));
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
            self.paths.insert(PathSet::one(path, self.kind));
        }
        self
    }

    // single alias, which does not correspond to any on-disk path
    pub fn alias(mut self, alias: &str) -> Self {
        // exceptional case for `Kind::Setup` because its `library`
        // and `compiler` options would otherwise naively match with
        // `compiler` and `library` folders respectively.
        assert!(
            self.kind == Kind::Setup || !self.builder.src.join(alias).exists(),
            "use `builder.path()` for real paths: {alias}"
        );
        self.paths.insert(PathSet::Set(
            std::iter::once(TaskPath { path: alias.into(), kind: Some(self.kind) }).collect(),
        ));
        self
    }

    /// single, non-aliased path
    ///
    /// Must be an on-disk path; use `alias` for names that do not correspond to on-disk paths.
    pub fn path(self, path: &str) -> Self {
        self.paths(&[path])
    }

    /// Multiple aliases for the same job.
    ///
    /// This differs from [`path`] in that multiple calls to path will end up calling `make_run`
    /// multiple times, whereas a single call to `paths` will only ever generate a single call to
    /// `make_run`.
    ///
    /// This is analogous to `all_krates`, although `all_krates` is gone now. Prefer [`path`] where possible.
    ///
    /// [`path`]: ShouldRun::path
    pub fn paths(mut self, paths: &[&str]) -> Self {
        let submodules_paths = self.builder.submodule_paths();

        self.paths.insert(PathSet::Set(
            paths
                .iter()
                .map(|p| {
                    // assert only if `p` isn't submodule
                    if !submodules_paths.iter().any(|sm_p| p.contains(sm_p)) {
                        assert!(
                            self.builder.src.join(p).exists(),
                            "`should_run.paths` should correspond to real on-disk paths - use `alias` if there is no relevant path: {p}"
                        );
                    }

                    TaskPath { path: p.into(), kind: Some(self.kind) }
                })
                .collect(),
        ));
        self
    }

    /// Handles individual files (not directories) within a test suite.
    fn is_suite_path(&self, requested_path: &Path) -> Option<&PathSet> {
        self.paths.iter().find(|pathset| match pathset {
            PathSet::Suite(suite) => requested_path.starts_with(&suite.path),
            PathSet::Set(_) => false,
        })
    }

    pub fn suite_path(mut self, suite: &str) -> Self {
        self.paths.insert(PathSet::Suite(TaskPath { path: suite.into(), kind: Some(self.kind) }));
        self
    }

    // allows being more explicit about why should_run in Step returns the value passed to it
    pub fn never(mut self) -> ShouldRun<'a> {
        self.paths.insert(PathSet::empty());
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
        kind: Kind,
    ) -> Vec<PathSet> {
        let mut sets = vec![];
        for pathset in &self.paths {
            let subset = pathset.intersection_removing_matches(paths, kind);
            if subset != PathSet::empty() {
                sets.push(subset);
            }
        }
        sets
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
    Suggest,
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
            Kind::Suggest => "suggest",
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
            Kind::Suggest => "Suggesting",
            Kind::Clippy => "Linting",
            Kind::Perf => "Profiling & benchmarking",
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
                builder.verbose(|| {
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

impl<'a> Builder<'a> {
    fn get_step_descriptions(kind: Kind) -> Vec<StepDescription> {
        macro_rules! describe {
            ($($rule:ty),+ $(,)?) => {{
                vec![$(StepDescription::from::<$rule>(kind)),+]
            }};
        }
        match kind {
            Kind::Build => describe!(
                compile::Std,
                compile::Rustc,
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
                check::CodegenBackend,
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
                check::FeaturesStatusDump,
                check::CoverageDump,
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
                test::Ui,
                test::Crashes,
                test::Coverage,
                test::MirOpt,
                test::Codegen,
                test::CodegenUnits,
                test::Assembly,
                test::Incremental,
                test::Debuginfo,
                test::UiFullDeps,
                test::Rustdoc,
                test::CoverageRunRustdoc,
                test::Pretty,
                test::CodegenCranelift,
                test::CodegenGCC,
                test::Crate,
                test::CrateLibrustc,
                test::CrateRustdoc,
                test::CrateRustdocJsonTypes,
                test::CrateBootstrap,
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
                // Run bootstrap close to the end as it's unlikely to fail
                test::Bootstrap,
                // Run run-make last, since these won't pass without make on Windows
                test::RunMake,
            ),
            Kind::Miri => describe!(test::Crate),
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
            Kind::Dist => describe!(
                dist::Docs,
                dist::RustcDocs,
                dist::JsonDocs,
                dist::Mingw,
                dist::Rustc,
                dist::CodegenBackend,
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
                dist::Bootstrap,
                dist::Extended,
                // It seems that PlainSourceTarball somehow changes how some of the tools
                // perceive their dependencies (see #93033) which would invalidate fingerprints
                // and force us to rebuild tools after vendoring dependencies.
                // To work around this, create the Tarball after building all the tools.
                dist::PlainSourceTarball,
                dist::BuildManifest,
                dist::ReproducibleArtifacts,
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
                install::Cargo,
                install::RustAnalyzer,
                install::Rustfmt,
                install::Clippy,
                install::Miri,
                install::LlvmTools,
                install::Src,
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
            ),
            Kind::Setup => {
                describe!(setup::Profile, setup::Hook, setup::Link, setup::Editor)
            }
            Kind::Clean => describe!(clean::CleanAll, clean::Rustc, clean::Std),
            Kind::Vendor => describe!(vendor::Vendor),
            // special-cased in Build::build()
            Kind::Format | Kind::Suggest | Kind::Perf => vec![],
            Kind::MiriTest | Kind::MiriSetup => unreachable!(),
        }
    }

    pub fn get_help(build: &Build, kind: Kind) -> Option<String> {
        let step_descriptions = Builder::get_step_descriptions(kind);
        if step_descriptions.is_empty() {
            return None;
        }

        let builder = Self::new_internal(build, kind, vec![]);
        let builder = &builder;
        // The "build" kind here is just a placeholder, it will be replaced with something else in
        // the following statement.
        let mut should_run = ShouldRun::new(builder, Kind::Build);
        for desc in step_descriptions {
            should_run.kind = desc.kind;
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

    fn new_internal(build: &Build, kind: Kind, paths: Vec<PathBuf>) -> Builder<'_> {
        Builder {
            build,
            top_stage: build.config.stage,
            kind,
            cache: Cache::new(),
            stack: RefCell::new(Vec::new()),
            time_spent_on_dependencies: Cell::new(Duration::new(0, 0)),
            paths,
            submodule_paths_cache: Default::default(),
        }
    }

    pub fn new(build: &Build) -> Builder<'_> {
        let paths = &build.config.paths;
        let (kind, paths) = match build.config.cmd {
            Subcommand::Build => (Kind::Build, &paths[..]),
            Subcommand::Check { .. } => (Kind::Check, &paths[..]),
            Subcommand::Clippy { .. } => (Kind::Clippy, &paths[..]),
            Subcommand::Fix => (Kind::Fix, &paths[..]),
            Subcommand::Doc { .. } => (Kind::Doc, &paths[..]),
            Subcommand::Test { .. } => (Kind::Test, &paths[..]),
            Subcommand::Miri { .. } => (Kind::Miri, &paths[..]),
            Subcommand::Bench { .. } => (Kind::Bench, &paths[..]),
            Subcommand::Dist => (Kind::Dist, &paths[..]),
            Subcommand::Install => (Kind::Install, &paths[..]),
            Subcommand::Run { .. } => (Kind::Run, &paths[..]),
            Subcommand::Clean { .. } => (Kind::Clean, &paths[..]),
            Subcommand::Format { .. } => (Kind::Format, &[][..]),
            Subcommand::Suggest { .. } => (Kind::Suggest, &[][..]),
            Subcommand::Setup { profile: ref path } => (
                Kind::Setup,
                path.as_ref().map_or([].as_slice(), |path| std::slice::from_ref(path)),
            ),
            Subcommand::Vendor { .. } => (Kind::Vendor, &paths[..]),
            Subcommand::Perf { .. } => (Kind::Perf, &paths[..]),
        };

        Self::new_internal(build, kind, paths.to_owned())
    }

    pub fn execute_cli(&self) {
        self.run_step_descriptions(&Builder::get_step_descriptions(self.kind), &self.paths);
    }

    pub fn default_doc(&self, paths: &[PathBuf]) {
        self.run_step_descriptions(&Builder::get_step_descriptions(Kind::Doc), paths);
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
        StepDescription::run(v, self, paths);
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

    /// Gets the paths to all of the compiler's codegen backends.
    fn codegen_backends(&self, compiler: Compiler) -> impl Iterator<Item = PathBuf> {
        fs::read_dir(self.sysroot_codegen_backends(compiler))
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
    }

    pub fn rustdoc(&self, compiler: Compiler) -> PathBuf {
        self.ensure(tool::Rustdoc { compiler }).tool_path
    }

    pub fn cargo_clippy_cmd(&self, run_compiler: Compiler) -> BootstrapCommand {
        if run_compiler.stage == 0 {
            let cargo_clippy = self
                .config
                .initial_cargo_clippy
                .clone()
                .unwrap_or_else(|| self.build.config.download_clippy());

            let mut cmd = command(cargo_clippy);
            cmd.env("CARGO", &self.initial_cargo);
            return cmd;
        }

        let _ =
            self.ensure(tool::Clippy { compiler: run_compiler, target: self.build.host_target });
        let cargo_clippy = self
            .ensure(tool::CargoClippy { compiler: run_compiler, target: self.build.host_target });
        let mut dylib_path = helpers::dylib_path();
        dylib_path.insert(0, self.sysroot(run_compiler).join("lib"));

        let mut cmd = command(cargo_clippy.tool_path);
        cmd.env(helpers::dylib_path_var(), env::join_paths(&dylib_path).unwrap());
        cmd.env("CARGO", &self.initial_cargo);
        cmd
    }

    pub fn cargo_miri_cmd(&self, run_compiler: Compiler) -> BootstrapCommand {
        assert!(run_compiler.stage > 0, "miri can not be invoked at stage 0");
        // Prepare the tools
        let miri =
            self.ensure(tool::Miri { compiler: run_compiler, target: self.build.host_target });
        let cargo_miri =
            self.ensure(tool::CargoMiri { compiler: run_compiler, target: self.build.host_target });
        // Invoke cargo-miri, make sure it can find miri and cargo.
        let mut cmd = command(cargo_miri.tool_path);
        cmd.env("MIRI", &miri.tool_path);
        cmd.env("CARGO", &self.initial_cargo);
        // Need to add the `run_compiler` libs. Those are the libs produces *by* `build_compiler`
        // in `tool::ToolBuild` step, so they match the Miri we just built. However this means they
        // are actually living one stage up, i.e. we are running `stage0-tools-bin/miri` with the
        // libraries in `stage1/lib`. This is an unfortunate off-by-1 caused (possibly) by the fact
        // that Miri doesn't have an "assemble" step like rustc does that would cross the stage boundary.
        // We can't use `add_rustc_lib_path` as that's a NOP on Windows but we do need these libraries
        // added to the PATH due to the stage mismatch.
        // Also see https://github.com/rust-lang/rust/pull/123192#issuecomment-2028901503.
        add_dylib_path(self.rustc_lib_paths(run_compiler), &mut cmd);
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
            .env("RUSTDOC_REAL", self.rustdoc(compiler))
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
    pub fn llvm_config(&self, target: TargetSelection) -> Option<PathBuf> {
        if self.config.llvm_enabled(target) && self.kind != Kind::Check && !self.config.dry_run() {
            let llvm::LlvmResult { llvm_config, .. } = self.ensure(llvm::Llvm { target });
            if llvm_config.is_file() {
                return Some(llvm_config);
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
                self.verbose_than(1, || println!("{}c {:?}", "  ".repeat(stack.len()), step));

                return out;
            }
            self.verbose_than(1, || println!("{}> {:?}", "  ".repeat(stack.len()), step));
            stack.push(Box::new(step.clone()));
        }

        #[cfg(feature = "build-metrics")]
        self.metrics.enter_step(&step, self);

        let (out, dur) = {
            let start = Instant::now();
            let zero = Duration::new(0, 0);
            let parent = self.time_spent_on_dependencies.replace(zero);
            let out = step.clone().run(self);
            let dur = start.elapsed();
            let deps = self.time_spent_on_dependencies.replace(parent + dur);
            (out, dur.saturating_sub(deps))
        };

        if self.config.print_step_timings && !self.config.dry_run() {
            let step_string = format!("{step:?}");
            let brace_index = step_string.find('{').unwrap_or(0);
            let type_string = type_name::<S>();
            println!(
                "[TIMING] {} {} -- {}.{:03}",
                &type_string.strip_prefix("bootstrap::").unwrap_or(type_string),
                &step_string[brace_index..],
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
        self.verbose_than(1, || println!("{}< {:?}", "  ".repeat(self.stack.borrow().len()), step));
        self.cache.put(step, out.clone());
        out
    }

    /// Ensure that a given step is built *only if it's supposed to be built by default*, returning
    /// its output. This will cache the step, so it's safe (and good!) to call this as often as
    /// needed to ensure that all dependencies are build.
    pub(crate) fn ensure_if_default<T, S: Step<Output = Option<T>>>(
        &'a self,
        step: S,
        kind: Kind,
    ) -> S::Output {
        let desc = StepDescription::from::<S>(kind);
        let should_run = (desc.should_run)(ShouldRun::new(self, desc.kind));

        // Avoid running steps contained in --skip
        for pathset in &should_run.paths {
            if desc.is_excluded(self, pathset) {
                return None;
            }
        }

        // Only execute if it's supposed to run as default
        if desc.default && should_run.is_really_default() { self.ensure(step) } else { None }
    }

    /// Checks if any of the "should_run" paths is in the `Builder` paths.
    pub(crate) fn was_invoked_explicitly<S: Step>(&'a self, kind: Kind) -> bool {
        let desc = StepDescription::from::<S>(kind);
        let should_run = (desc.should_run)(ShouldRun::new(self, desc.kind));

        for path in &self.paths {
            if should_run.paths.iter().any(|s| s.has(path, desc.kind))
                && !desc.is_excluded(
                    self,
                    &PathSet::Suite(TaskPath { path: path.clone(), kind: Some(desc.kind) }),
                )
            {
                return true;
            }
        }

        false
    }

    pub(crate) fn maybe_open_in_browser<S: Step>(&self, path: impl AsRef<Path>) {
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

impl<'a> AsRef<ExecutionContext> for Builder<'a> {
    fn as_ref(&self) -> &ExecutionContext {
        self.exec_ctx()
    }
}
