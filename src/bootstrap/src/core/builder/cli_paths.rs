//! Various pieces of code for dealing with "paths" passed to bootstrap on the
//! command-line, extracted from `core/builder/mod.rs` because that file is
//! large and hard to navigate.

use std::fmt::{self, Debug};
use std::path::PathBuf;

use crate::core::builder::{Builder, CommandLineStepDescription, Kind, PathSet, ShouldRun};

#[cfg(test)]
mod tests;

#[derive(Clone, PartialEq)]
pub(crate) struct CLIStepPath {
    pub(crate) path: PathBuf,
    pub(crate) will_be_executed: bool,
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

/// Combines a [`CommandLineStepDescription`] with its corresponding [`ShouldRun`].
struct StepExtra<'a> {
    desc: &'a CommandLineStepDescription,
    should_run: ShouldRun<'a>,
}

struct StepToRun<'a> {
    sort_index: usize,
    desc: &'a CommandLineStepDescription,
    pathsets: Vec<PathSet>,
}

pub(crate) fn match_paths_to_steps_and_run(
    builder: &Builder<'_>,
    step_descs: &[CommandLineStepDescription],
    paths: &[PathBuf],
) {
    // Obtain `ShouldRun` information for each step, so that we know which
    // paths to match it against.
    let steps = step_descs
        .iter()
        .map(|desc| StepExtra { desc, should_run: (desc.should_run)(ShouldRun::new(builder)) })
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
            if (desc.is_default_step_fn)(builder) {
                let default_pathsets = should_run.default_pathsets();
                desc.maybe_run(builder, default_pathsets);
            }
        }
    }

    // Command-line paths are interpreted relative to the repository root
    // (not the current working directory).
    //
    // If the user or shell passed an absolute path, try to strip off the
    // repository root, to match the paths registered by command-line steps.
    //
    // E.g. `/home/ferris/rust/tests/ui/asm/cfg.rs` => `tests/ui/asm/cfg.rs`
    let mut paths = paths
        .iter()
        .map(|path| {
            if path.is_absolute()
                && path.exists()
                && let Ok(relative) = path.strip_prefix(&builder.src)
            {
                relative
            } else {
                path
            }
        })
        .map(|p| p.to_owned())
        .collect::<Vec<_>>();

    // If any absolute paths couldn't be made relative, stop now and report them.
    let bad_abs_paths = paths.iter().filter(|path| path.is_absolute()).collect::<Vec<_>>();
    if !bad_abs_paths.is_empty() {
        eprintln!("ERROR: failed to resolve absolute paths: {bad_abs_paths:#?}");
        crate::exit!(1);
    }

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
        let pathsets = should_run.pathsets_for_paths_flagging_matches(&mut paths);

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
