//! Runs rustfmt on the repository.

use crate::Build;
use build_helper::{output, t};
use ignore::WalkBuilder;
use std::path::Path;
use std::process::{Command, Stdio};

fn rustfmt(src: &Path, rustfmt: &Path, path: &Path, check: bool) {
    let mut cmd = Command::new(&rustfmt);
    // avoid the submodule config paths from coming into play,
    // we only allow a single global config for the workspace for now
    cmd.arg("--config-path").arg(&src.canonicalize().unwrap());
    cmd.arg("--edition").arg("2018");
    cmd.arg("--unstable-features");
    cmd.arg("--skip-children");
    if check {
        cmd.arg("--check");
    }
    cmd.arg(&path);
    let cmd_debug = format!("{:?}", cmd);
    let status = cmd.status().expect("executing rustfmt");
    if !status.success() {
        eprintln!(
            "Running `{}` failed.\nIf you're running `tidy`, \
            try again with `--bless`. Or, if you just want to format \
            code, run `./x.py fmt` instead.",
            cmd_debug,
        );
        std::process::exit(1);
    }
}

#[derive(serde::Deserialize)]
struct RustfmtConfig {
    ignore: Vec<String>,
}

pub fn format(build: &Build, check: bool) {
    if build.config.dry_run {
        return;
    }
    let mut builder = ignore::types::TypesBuilder::new();
    builder.add_defaults();
    builder.select("rust");
    let matcher = builder.build().unwrap();
    let rustfmt_config = build.src.join("rustfmt.toml");
    if !rustfmt_config.exists() {
        eprintln!("Not running formatting checks; rustfmt.toml does not exist.");
        eprintln!("This may happen in distributed tarballs.");
        return;
    }
    let rustfmt_config = t!(std::fs::read_to_string(&rustfmt_config));
    let rustfmt_config: RustfmtConfig = t!(toml::from_str(&rustfmt_config));
    let mut ignore_fmt = ignore::overrides::OverrideBuilder::new(&build.src);
    for ignore in rustfmt_config.ignore {
        ignore_fmt.add(&format!("!{}", ignore)).expect(&ignore);
    }
    let git_available = match Command::new("git")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(status) => status.success(),
        Err(_) => false,
    };
    if git_available {
        let in_working_tree = match Command::new("git")
            .arg("rev-parse")
            .arg("--is-inside-work-tree")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
        {
            Ok(status) => status.success(),
            Err(_) => false,
        };
        if in_working_tree {
            let untracked_paths_output = output(
                Command::new("git")
                    .arg("status")
                    .arg("--porcelain")
                    .arg("--untracked-files=normal"),
            );
            let untracked_paths = untracked_paths_output
                .lines()
                .filter(|entry| entry.starts_with("??"))
                .map(|entry| {
                    entry.split(' ').nth(1).expect("every git status entry should list a path")
                });
            for untracked_path in untracked_paths {
                eprintln!("skip untracked path {} during rustfmt invocations", untracked_path);
                ignore_fmt.add(&format!("!{}", untracked_path)).expect(&untracked_path);
            }
        } else {
            eprintln!("Not in git tree. Skipping git-aware format checks");
        }
    } else {
        eprintln!("Could not find usable git. Skipping git-aware format checks");
    }
    let ignore_fmt = ignore_fmt.build().unwrap();

    let rustfmt_path = build.config.initial_rustfmt.as_ref().unwrap_or_else(|| {
        eprintln!("./x.py fmt is not supported on this channel");
        std::process::exit(1);
    });
    let src = build.src.clone();
    let walker = WalkBuilder::new(&build.src).types(matcher).overrides(ignore_fmt).build_parallel();
    walker.run(|| {
        let src = src.clone();
        let rustfmt_path = rustfmt_path.clone();
        Box::new(move |entry| {
            let entry = t!(entry);
            if entry.file_type().map_or(false, |t| t.is_file()) {
                rustfmt(&src, &rustfmt_path, &entry.path(), check);
            }
            ignore::WalkState::Continue
        })
    });
}
