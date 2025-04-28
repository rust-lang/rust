//! Attempt to magically identify good tests to run

use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;

use crate::core::build_steps::tool::Tool;
use crate::core::builder::Builder;

/// Suggests a list of possible `x.py` commands to run based on modified files in branch.
pub fn suggest(builder: &Builder<'_>, run: bool) {
    let git_config = builder.config.git_config();
    let suggestions = builder
        .tool_cmd(Tool::SuggestTests)
        .env("SUGGEST_TESTS_NIGHTLY_BRANCH", git_config.nightly_branch)
        .env("SUGGEST_TESTS_MERGE_COMMIT_EMAIL", git_config.git_merge_commit_email)
        .run_capture_stdout(builder)
        .stdout();

    let suggestions = suggestions
        .lines()
        .map(|line| {
            let mut sections = line.split_ascii_whitespace();

            // this code expects one suggestion per line in the following format:
            // <x_subcommand> {some number of flags} [optional stage number]
            let cmd = sections.next().unwrap();
            let stage = sections.next_back().and_then(|s| str::parse(s).ok());
            let paths: Vec<PathBuf> = sections.map(|p| PathBuf::from_str(p).unwrap()).collect();

            (cmd, stage, paths)
        })
        .collect::<Vec<_>>();

    if !suggestions.is_empty() {
        println!("==== SUGGESTIONS ====");
        for sug in &suggestions {
            print!("x {} ", sug.0);
            if let Some(stage) = sug.1 {
                print!("--stage {stage} ");
            }

            for path in &sug.2 {
                print!("{} ", path.display());
            }
            println!();
        }
        println!("=====================");
    } else {
        println!("No suggestions found!");
        return;
    }

    if run {
        for sug in suggestions {
            let mut build: crate::Build = builder.build.clone();
            build.config.paths = sug.2;
            build.config.cmd = crate::core::config::flags::Flags::parse_from(["x.py", sug.0]).cmd;
            if let Some(stage) = sug.1 {
                build.config.stage = stage;
            }
            build.build();
        }
    } else {
        println!("HELP: consider using the `--run` flag to automatically run suggested tests");
    }
}
