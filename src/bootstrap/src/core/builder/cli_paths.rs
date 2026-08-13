//! Various pieces of code for dealing with "paths" passed to bootstrap on the
//! command-line, extracted from `core/builder/mod.rs` because that file is
//! large and hard to navigate.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::PathBuf;

use crate::core::builder::{Builder, CommandLineStepDescription, Kind, PathSet, ShouldRun};
use crate::utils::helpers;

#[cfg(test)]
mod tests;

/// Combines a [`CommandLineStepDescription`] with its corresponding [`ShouldRun`].
struct StepExtra<'a> {
    desc: &'a CommandLineStepDescription,
    should_run: ShouldRun<'a>,
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
        helpers::exit_process(1);
    }

    // sanity checks on rules
    for StepExtra { desc, should_run } in &steps {
        assert!(!should_run.paths.is_empty(), "{:?} should have at least one pathset", desc.name);
    }

    // Run default steps if appropriate.
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
    //
    // It is also possible that someone passed a relative path starting with . or ..
    // In that case, we have to remove that path prefix.
    let paths = paths
        .iter()
        .map(|path| {
            // Here we "launder" the path through builder.src, to normalize relative path prefixes
            // so ./tests/foo becomes just tests/foo
            let path = if path.is_relative() {
                builder
                    .src
                    .join(path)
                    .strip_prefix(&builder.src)
                    .expect("Cannot strip src path prefix")
                    .to_path_buf()
            } else {
                path.to_path_buf()
            };

            if path.is_absolute()
                && path.exists()
                && let Ok(relative) = path.strip_prefix(&builder.src)
            {
                relative.to_path_buf()
            } else {
                path
            }
        })
        .collect::<Vec<_>>();

    // If any absolute paths couldn't be made relative, stop now and report them.
    let bad_abs_paths = paths.iter().filter(|path| path.is_absolute()).collect::<Vec<_>>();
    if !bad_abs_paths.is_empty() {
        eprintln!(
            "ERROR: the following paths do not exist on disk or point outside the source directory: {bad_abs_paths:#?}"
        );
        helpers::exit_process(1);
    }

    // When matching selectors to steps, we want to balance two conflicting goals:
    // - Ideally, steps should run in the order specified by command-line arguments.
    // - A selected step should be invoked only once, not multiple times.
    //
    // We therefore build up:
    // - An ordered list of steps to run, each represented by its index in `steps`.
    // - For each step (by index), the list of its anchors that were matched.
    let mut step_queue = Vec::<usize>::with_capacity(paths.len());
    let mut step_anchors = HashMap::<usize, Vec<&PathSet>>::with_capacity(steps.len());
    let mut unmatched_paths = vec![];

    // For each command-line selector, enqueue the steps that it matches.
    for path in &paths {
        let mut path_matched = false;

        for (step_ix, step) in steps.iter().enumerate() {
            let matched_anchors = step
                .should_run
                .paths
                .iter()
                .filter(|anchor| {
                    // The extra `starts_with` here allows an argument like
                    // `tests/ui/asm/cfg.rs` to select the suite anchor `tests/ui`.
                    anchor.has(path)
                        || matches!(anchor, PathSet::Suite(suite) if path.starts_with(&suite.path))
                })
                .collect::<Vec<_>>();

            if !matched_anchors.is_empty() {
                step_queue.push(step_ix);
                step_anchors.entry(step_ix).or_default().extend(matched_anchors);
                path_matched = true;
            }
        }

        if !path_matched {
            unmatched_paths.push(path);
        }
    }

    if !unmatched_paths.is_empty() {
        eprintln!("ERROR: no `{}` rules matched {unmatched_paths:?}", builder.kind.as_str());
        eprintln!(
            "HELP: run `x.py {} --help --verbose` to show a list of available paths",
            builder.kind.as_str()
        );
        eprintln!(
            "NOTE: if you are adding a new Step to bootstrap itself, make sure you register it with `describe!`"
        );
        helpers::exit_process(1);
    }

    fn dedup_vec<T: Copy + Eq + Hash>(vec: &mut Vec<T>) {
        let mut seen = HashSet::<T>::with_capacity(vec.len());
        vec.retain(|&x| seen.insert(x));
    }

    // Deduplicate the queue of steps to run, and the list of anchors to run for each step.
    dedup_vec(&mut step_queue);
    for anchors in step_anchors.values_mut() {
        dedup_vec(anchors);
    }

    // Run the steps that were selected, in (roughly) command-line order.
    // For each step, pass all of its matched anchors, regardless of position.
    for &step_ix in &step_queue {
        let step = &steps[step_ix];
        let anchors = step_anchors[&step_ix].iter().map(|p| PathSet::clone(p)).collect::<Vec<_>>();
        step.desc.maybe_run(builder, anchors);
    }
}
