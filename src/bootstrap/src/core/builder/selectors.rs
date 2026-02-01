//! Various pieces of code for dealing with "paths" passed to bootstrap on the
//! command-line, extracted from `core/builder/mod.rs` because that file is
//! large and hard to navigate.

use std::collections::BTreeSet;
use std::fmt::{self, Debug, Write};
use std::path::{Path, PathBuf};

use crate::core::builder::{Builder, CargoSubcommand, RunConfig, Step, StepDescription};
use crate::core::config::DryRun;
use crate::{Build, Crate, t};

#[cfg(test)]
mod tests;

pub(crate) const PATH_REMAP: &[(&str, &[&str])] = &[
    // Make `x test tests` function the same as `x t tests/*`
    (
        "tests",
        &[
            // tidy-alphabetical-start
            "tests/assembly-llvm",
            "tests/build-std",
            "tests/codegen-llvm",
            "tests/codegen-units",
            "tests/coverage",
            "tests/coverage-run-rustdoc",
            "tests/crashes",
            "tests/debuginfo",
            "tests/incremental",
            "tests/mir-opt",
            "tests/pretty",
            "tests/run-make",
            "tests/run-make-cargo",
            "tests/rustdoc-gui",
            "tests/rustdoc-html",
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

pub(crate) fn remap_paths(paths: &mut Vec<PathBuf>) {
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
pub(crate) struct CLIStepPath {
    pub(crate) path: PathBuf,
    pub(crate) will_be_executed: bool,
}

#[cfg(test)]
impl CLIStepPath {
    pub(crate) fn will_be_executed(mut self, will_be_executed: bool) -> Self {
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

/// Combines a `StepDescription` with its corresponding `ShouldRun`.
struct StepExtra<'a> {
    desc: &'a StepDescription,
    should_run: ShouldRun<'a>,
}

struct StepToRun<'a> {
    sort_index: usize,
    desc: &'a StepDescription,
    pathsets: Vec<StepSelectors>,
}

pub(crate) fn match_paths_to_steps_and_run(
    builder: &Builder<'_>,
    step_descs: &[StepDescription],
    paths: &[PathBuf],
) {
    // Obtain `ShouldRun` information for each step, so that we know which
    // paths to match it against.
    let steps = step_descs
        .iter()
        .map(|desc| StepExtra {
            desc,
            should_run: (desc.should_run)(ShouldRun::new(builder, desc.cargo_cmd)),
        })
        .collect::<Vec<_>>();

    // FIXME(Zalathar): This particular check isn't related to path-to-step
    // matching, and should probably be hoisted to somewhere much earlier.
    if builder.download_rustc()
        && (builder.cargo_cmd == CargoSubcommand::Dist
            || builder.cargo_cmd == CargoSubcommand::Install)
    {
        eprintln!(
            "ERROR: '{}' subcommand is incompatible with `rust.download-rustc`.",
            builder.cargo_cmd.as_str()
        );
        crate::exit!(1);
    }

    // sanity checks on rules
    for StepExtra { desc, should_run } in &steps {
        assert!(!should_run.paths.is_empty(), "{:?} should have at least one pathset", desc.name);
    }

    if paths.is_empty() || builder.config.include_default_paths {
        for StepExtra { desc, should_run } in &steps {
            if (desc.is_default_step_fn)(builder) {
                desc.maybe_run(builder, should_run.paths.iter().cloned().collect());
            }
        }
    }

    // Attempt to resolve paths to be relative to the builder source directory.
    let mut paths: Vec<PathBuf> = paths
        .iter()
        .map(|original_path| {
            let mut path = original_path.clone();

            // Someone could run `x <cmd> <path>` from a different repository than the source
            // directory.
            // In that case, we should not try to resolve the paths relative to the working
            // directory, but rather relative to the source directory.
            // So we forcefully "relocate" the path to the source directory here.
            if !path.is_absolute() {
                path = builder.src.join(path);
            }

            // If the path does not exist, it may represent the name of a Step, such as `tidy` in `x test tidy`
            if !path.exists() {
                // Use the original path here
                return original_path.clone();
            }

            // Make the path absolute, strip the prefix, and convert to a PathBuf.
            match std::path::absolute(&path) {
                Ok(p) => p.strip_prefix(&builder.src).unwrap_or(&p).to_path_buf(),
                Err(e) => {
                    eprintln!("ERROR: {e:?}");
                    panic!("Due to the above error, failed to resolve path: {path:?}");
                }
            }
        })
        .collect();

    remap_paths(&mut paths);

    // Handle all test suite paths.
    // (This is separate from the loop below to avoid having to handle multiple paths in `is_suite_path` somehow.)
    paths.retain(|path| {
        for StepExtra { desc, should_run } in &steps {
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

    // Before actually running (non-suite) steps, collect them into a list of structs
    // so that we can then sort the list to preserve CLI order as much as possible.
    let mut steps_to_run = vec![];

    for StepExtra { desc, should_run } in &steps {
        let pathsets = should_run.pathset_for_paths_removing_matches(&mut paths, desc.cargo_cmd);

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

        steps_to_run.push(StepToRun { sort_index: closest_index, desc, pathsets });
    }

    // Sort the steps before running them to respect the CLI order.
    steps_to_run.sort_by_key(|step| step.sort_index);

    // Handle all PathSets.
    for StepToRun { sort_index: _, desc, pathsets } in steps_to_run {
        if !pathsets.is_empty() {
            desc.maybe_run(builder, pathsets);
        }
    }

    paths.retain(|p| !p.will_be_executed);

    if !paths.is_empty() {
        eprintln!("ERROR: no `{}` rules matched {:?}", builder.cargo_cmd.as_str(), paths);
        eprintln!(
            "HELP: run `x.py {} --help --verbose` to show a list of available paths",
            builder.cargo_cmd.as_str()
        );
        eprintln!(
            "NOTE: if you are adding a new Step to bootstrap itself, make sure you register it with `describe!`"
        );
        crate::exit!(1);
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

impl RunConfig<'_> {
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
        if let Some(p_kind) = &p.cargo_cmd {
            check_path() && *p_kind == module
        } else {
            check_path()
        }
    }

    /// Return all `TaskPath`s in `Self` that contain any of the `needles`, removing the
    /// matched needles.
    ///
    /// This is used for `StepDescription::krate`, which passes all matching crates at once to
    /// `Step::make_run`, rather than calling it many times with a single crate.
    /// See `tests.rs` for examples.
    fn intersection_removing_matches(
        &self,
        needles: &mut [CLIStepPath],
        module: CargoSubcommand,
    ) -> StepSelectors {
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
            StepSelectors::Alias(set) => {
                StepSelectors::Alias(set.iter().filter(|&p| check(p)).cloned().collect())
            }
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
            StepSelectors::TestSuite(_) => {
                unreachable!("called assert_single_path on a Suite path")
            }
        }
    }
}

impl StepDescription {
    pub(crate) fn from<S: Step>(cargo_cmd: CargoSubcommand) -> StepDescription {
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
            std::iter::once(StepSelection { path: alias.into(), cargo_cmd: Some(self.cargo_cmd) })
                .collect(),
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
        self.paths.insert(StepSelectors::TestSuite(StepSelection {
            path: suite.into(),
            cargo_cmd: Some(self.cargo_cmd),
        }));
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

impl<'a> Builder<'a> {
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
                    &StepSelectors::TestSuite(StepSelection {
                        path: path.clone(),
                        cargo_cmd: Some(desc.cargo_cmd),
                    }),
                )
            {
                return true;
            }
        }

        false
    }
}
