//! Runs rustfmt on the repository.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::mpsc::SyncSender;

use build_helper::git::get_git_modified_files;
use ignore::WalkBuilder;

use crate::core::builder::{Builder, Kind};
use crate::utils::build_stamp::BuildStamp;
use crate::utils::exec::command;
use crate::utils::helpers::{self, t};

#[must_use]
enum RustfmtStatus {
    InProgress,
    Ok,
    Failed,
}

fn rustfmt(
    src: &Path,
    rustfmt: &Path,
    paths: &[PathBuf],
    check: bool,
) -> impl FnMut(bool) -> RustfmtStatus + use<> {
    let mut cmd = Command::new(rustfmt);
    // Avoid the submodule config paths from coming into play. We only allow a single global config
    // for the workspace for now.
    cmd.arg("--config-path").arg(src.canonicalize().unwrap());
    cmd.arg("--edition").arg("2024");
    cmd.arg("--unstable-features");
    cmd.arg("--skip-children");
    if check {
        cmd.arg("--check");
    }
    cmd.args(paths);
    let mut cmd = cmd.spawn().expect("running rustfmt");
    // Poor man's async: return a closure that might wait for rustfmt's completion (depending on
    // the value of the `block` argument).
    move |block: bool| -> RustfmtStatus {
        let status = if !block {
            match cmd.try_wait() {
                Ok(Some(status)) => Ok(status),
                Ok(None) => return RustfmtStatus::InProgress,
                Err(err) => Err(err),
            }
        } else {
            cmd.wait()
        };
        if status.unwrap().success() { RustfmtStatus::Ok } else { RustfmtStatus::Failed }
    }
}

fn get_rustfmt_version(build: &Builder<'_>) -> Option<(String, BuildStamp)> {
    let stamp_file = BuildStamp::new(&build.out).with_prefix("rustfmt");

    let mut cmd = command(build.config.initial_rustfmt.as_ref()?);
    cmd.arg("--version");

    let output = cmd.allow_failure().run_capture(build);
    if output.is_failure() {
        return None;
    }
    Some((output.stdout(), stamp_file))
}

/// Return whether the format cache can be reused.
fn verify_rustfmt_version(build: &Builder<'_>) -> bool {
    let Some((version, stamp_file)) = get_rustfmt_version(build) else {
        return false;
    };
    stamp_file.add_stamp(version).is_up_to_date()
}

/// Updates the last rustfmt version used.
fn update_rustfmt_version(build: &Builder<'_>) {
    let Some((version, stamp_file)) = get_rustfmt_version(build) else {
        return;
    };

    t!(stamp_file.add_stamp(version).write());
}

/// Returns the Rust files modified between the last merge commit and what is now on the disk.
/// Does not include removed files.
///
/// Returns `None` if all files should be formatted.
fn get_modified_rs_files(build: &Builder<'_>) -> Result<Option<Vec<String>>, String> {
    // In CI `get_git_modified_files` returns something different to normal environment.
    // This shouldn't be called in CI anyway.
    assert!(!build.config.is_running_on_ci);

    if !verify_rustfmt_version(build) {
        return Ok(None);
    }

    get_git_modified_files(&build.config.git_config(), Some(&build.config.src), &["rs"]).map(Some)
}

#[derive(serde_derive::Deserialize)]
struct RustfmtConfig {
    ignore: Vec<String>,
}

// Prints output describing a collection of paths, with lines such as "formatted modified file
// foo/bar/baz" or "skipped 20 untracked files".
fn print_paths(verb: &str, adjective: Option<&str>, paths: &[String]) {
    let len = paths.len();
    let adjective =
        if let Some(adjective) = adjective { format!("{adjective} ") } else { String::new() };
    if len <= 10 {
        for path in paths {
            println!("fmt: {verb} {adjective}file {path}");
        }
    } else {
        println!("fmt: {verb} {len} {adjective}files");
    }
}

pub fn format(build: &Builder<'_>, check: bool, all: bool, paths: &[PathBuf]) {
    if build.kind == Kind::Format && build.top_stage != 0 {
        eprintln!("ERROR: `x fmt` only supports stage 0.");
        eprintln!("HELP: Use `x run rustfmt` to run in-tree rustfmt.");
        crate::exit!(1);
    }

    if !paths.is_empty() {
        eprintln!(
            "fmt error: path arguments are no longer accepted; use `--all` to format everything"
        );
        crate::exit!(1);
    };
    if build.config.dry_run() {
        return;
    }

    // By default, we only check modified files locally to speed up runtime. Exceptions are if
    // `--all` is specified or we are in CI. We check all files in CI to avoid bugs in
    // `get_modified_rs_files` letting regressions slip through; we also care about CI time less
    // since this is still very fast compared to building the compiler.
    let all = all || build.config.is_running_on_ci;

    let mut builder = ignore::types::TypesBuilder::new();
    builder.add_defaults();
    builder.select("rust");
    let matcher = builder.build().unwrap();
    let rustfmt_config = build.src.join("rustfmt.toml");
    if !rustfmt_config.exists() {
        eprintln!("fmt error: Not running formatting checks; rustfmt.toml does not exist.");
        eprintln!("fmt error: This may happen in distributed tarballs.");
        return;
    }
    let rustfmt_config = t!(std::fs::read_to_string(&rustfmt_config));
    let rustfmt_config: RustfmtConfig = t!(toml::from_str(&rustfmt_config));
    let mut override_builder = ignore::overrides::OverrideBuilder::new(&build.src);
    for ignore in rustfmt_config.ignore {
        if ignore.starts_with('!') {
            // A `!`-prefixed entry could be added as a whitelisted entry in `override_builder`,
            // i.e. strip the `!` prefix. But as soon as whitelisted entries are added, an
            // `OverrideBuilder` will only traverse those whitelisted entries, and won't traverse
            // any files that aren't explicitly mentioned. No bueno! Maybe there's a way to combine
            // explicit whitelisted entries and traversal of unmentioned files, but for now just
            // forbid such entries.
            eprintln!("fmt error: `!`-prefixed entries are not supported in rustfmt.toml, sorry");
            crate::exit!(1);
        } else {
            override_builder.add(&format!("!{ignore}")).expect(&ignore);
        }
    }
    let git_available =
        helpers::git(None).allow_failure().arg("--version").run_capture(build).is_success();

    let mut adjective = None;
    if git_available {
        let in_working_tree = helpers::git(Some(&build.src))
            .allow_failure()
            .arg("rev-parse")
            .arg("--is-inside-work-tree")
            .run_capture(build)
            .is_success();
        if in_working_tree {
            let untracked_paths_output = helpers::git(Some(&build.src))
                .arg("status")
                .arg("--porcelain")
                .arg("-z")
                .arg("--untracked-files=normal")
                .run_capture_stdout(build)
                .stdout();
            let untracked_paths: Vec<_> = untracked_paths_output
                .split_terminator('\0')
                .filter_map(
                    |entry| entry.strip_prefix("?? "), // returns None if the prefix doesn't match
                )
                .map(|x| x.to_string())
                .collect();
            print_paths("skipped", Some("untracked"), &untracked_paths);

            for untracked_path in untracked_paths {
                // The leading `/` makes it an exact match against the
                // repository root, rather than a glob. Without that, if you
                // have `foo.rs` in the repository root it will also match
                // against anything like `compiler/rustc_foo/src/foo.rs`,
                // preventing the latter from being formatted.
                override_builder.add(&format!("!/{untracked_path}")).expect(&untracked_path);
            }
            if !all {
                adjective = Some("modified");
                match get_modified_rs_files(build) {
                    Ok(Some(files)) => {
                        if files.is_empty() {
                            println!("fmt info: No modified files detected for formatting.");
                            return;
                        }

                        for file in files {
                            override_builder.add(&format!("/{file}")).expect(&file);
                        }
                    }
                    Ok(None) => {
                        // NOTE: `Ok(None)` signifies that we need to format all files.
                        // The tricky part here is that if `override_builder` isn't given any white
                        // list files (i.e. files to be formatted, added without leading `!`), it
                        // will instead look for *all* files. So, by doing nothing here, we are
                        // actually making it so we format all files.
                    }
                    Err(err) => {
                        eprintln!("fmt warning: Something went wrong running git commands:");
                        eprintln!("fmt warning: {err}");
                        eprintln!("fmt warning: Falling back to formatting all files.");
                    }
                }
            }
        } else {
            eprintln!("fmt: warning: Not in git tree. Skipping git-aware format checks");
        }
    } else {
        eprintln!("fmt: warning: Could not find usable git. Skipping git-aware format checks");
    }

    let override_ = override_builder.build().unwrap(); // `override` is a reserved keyword

    let rustfmt_path = build.config.initial_rustfmt.clone().unwrap_or_else(|| {
        eprintln!("fmt error: `x fmt` is not supported on this channel");
        crate::exit!(1);
    });
    assert!(rustfmt_path.exists(), "{}", rustfmt_path.display());
    let src = build.src.clone();
    let (tx, rx): (SyncSender<PathBuf>, _) = std::sync::mpsc::sync_channel(128);
    let walker = WalkBuilder::new(src.clone()).types(matcher).overrides(override_).build_parallel();

    // There is a lot of blocking involved in spawning a child process and reading files to format.
    // Spawn more processes than available concurrency to keep the CPU busy.
    let max_processes = build.jobs() as usize * 2;

    // Spawn child processes on a separate thread so we can batch entries we have received from
    // ignore.
    let thread = std::thread::spawn(move || {
        let mut result = Ok(());

        let mut children = VecDeque::new();
        while let Ok(path) = rx.recv() {
            // Try getting more paths from the channel to amortize the overhead of spawning
            // processes.
            let paths: Vec<_> = rx.try_iter().take(63).chain(std::iter::once(path)).collect();

            let child = rustfmt(&src, &rustfmt_path, paths.as_slice(), check);
            children.push_back(child);

            // Poll completion before waiting.
            for i in (0..children.len()).rev() {
                match children[i](false) {
                    RustfmtStatus::InProgress => {}
                    RustfmtStatus::Failed => {
                        result = Err(());
                        children.swap_remove_back(i);
                        break;
                    }
                    RustfmtStatus::Ok => {
                        children.swap_remove_back(i);
                        break;
                    }
                }
            }

            if children.len() >= max_processes {
                // Await oldest child.
                match children.pop_front().unwrap()(true) {
                    RustfmtStatus::InProgress | RustfmtStatus::Ok => {}
                    RustfmtStatus::Failed => result = Err(()),
                }
            }
        }

        // Await remaining children.
        for mut child in children {
            match child(true) {
                RustfmtStatus::InProgress | RustfmtStatus::Ok => {}
                RustfmtStatus::Failed => result = Err(()),
            }
        }

        result
    });

    let formatted_paths = Mutex::new(Vec::new());
    let formatted_paths_ref = &formatted_paths;
    walker.run(|| {
        let tx = tx.clone();
        Box::new(move |entry| {
            let cwd = std::env::current_dir();
            let entry = t!(entry);
            if entry.file_type().is_some_and(|t| t.is_file()) {
                formatted_paths_ref.lock().unwrap().push({
                    // `into_path` produces an absolute path. Try to strip `cwd` to get a shorter
                    // relative path.
                    let mut path = entry.clone().into_path();
                    if let Ok(cwd) = cwd {
                        if let Ok(path2) = path.strip_prefix(cwd) {
                            path = path2.to_path_buf();
                        }
                    }
                    path.display().to_string()
                });
                t!(tx.send(entry.into_path()));
            }
            ignore::WalkState::Continue
        })
    });
    let mut paths = formatted_paths.into_inner().unwrap();
    paths.sort();
    print_paths(if check { "checked" } else { "formatted" }, adjective, &paths);

    drop(tx);

    let result = thread.join().unwrap();

    if result.is_err() {
        crate::exit!(1);
    }

    // Update `build/.rustfmt-stamp`, allowing this code to ignore files which have not been changed
    // since last merge.
    //
    // NOTE: Because of the exit above, this is only reachable if formatting / format checking
    // succeeded. So we are not commiting the version if formatting was not good.
    update_rustfmt_version(build);
}
