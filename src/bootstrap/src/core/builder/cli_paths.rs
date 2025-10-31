//! Various pieces of code for dealing with "paths" passed to bootstrap on the
//! command-line, extracted from `core/builder/mod.rs` because that file is
//! large and hard to navigate.

use std::fmt::{self, Debug};
use std::path::PathBuf;

use crate::core::builder::{Builder, Kind, PathSet, ShouldRun, StepDescription};

pub(crate) const PATH_REMAP: &[(&str, &[&str])] = &[
    // bootstrap.toml uses `rust-analyzer-proc-macro-srv`, but the
    // actual path is `proc-macro-srv-cli`
    ("rust-analyzer-proc-macro-srv", &["src/tools/rust-analyzer/crates/proc-macro-srv-cli"]),
    // Make `x test tests` function the same as `x t tests/*`
    (
        "tests",
        &[
            // tidy-alphabetical-start
            "tests/assembly-llvm",
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
    pathsets: Vec<PathSet>,
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
            should_run: (desc.should_run)(ShouldRun::new(builder, desc.kind)),
        })
        .collect::<Vec<_>>();

    // FIXME(Zalathar): This particular check isn't related to path-to-step
    // matching, and should probably be hoisted to somewhere much earlier.
    if builder.download_rustc() && (builder.kind == Kind::Dist || builder.kind == Kind::Install) {
        eprintln!(
            "ERROR: '{}' subcommand is incompatible with `rust.download-rustc`.",
            builder.kind.as_str()
        );
        crate::exit!(1);
    }

    // sanity checks on rules
    for StepExtra { desc, should_run } in &steps {
        assert!(!should_run.paths.is_empty(), "{:?} should have at least one pathset", desc.name);
    }

    if paths.is_empty() || builder.config.include_default_paths {
        for StepExtra { desc, should_run } in &steps {
            if desc.default && should_run.is_really_default() {
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
