#![cfg_attr(feature = "build-metrics", allow(unused))]

use std::str::FromStr;

use std::path::PathBuf;

use clap::Parser;

use crate::{builder::Builder, tool::Tool};

/// Suggests a list of possible `x.py` commands to run based on modified files in branch.
pub fn suggest(builder: &Builder<'_>, run: bool) {
    let suggestions =
        builder.tool_cmd(Tool::SuggestTests).output().expect("failed to run `suggest-tests` tool");

    if !suggestions.status.success() {
        println!("failed to run `suggest-tests` tool ({})", suggestions.status);
        println!(
            "`suggest_tests` stdout:\n{}`suggest_tests` stderr:\n{}",
            String::from_utf8(suggestions.stdout).unwrap(),
            String::from_utf8(suggestions.stderr).unwrap()
        );
        panic!("failed to run `suggest-tests`");
    }

    let suggestions = String::from_utf8(suggestions.stdout).unwrap();
    let suggestions = suggestions
        .lines()
        .map(|line| {
            let mut sections = line.split_ascii_whitespace();

            // this code expects one suggestion per line in the following format:
            // <x_subcommand> {some number of flags} [optional stage number]
            let cmd = sections.next().unwrap();
            let stage = sections.next_back().map(|s| str::parse(s).ok()).flatten();
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
            build.config.cmd = crate::flags::Flags::parse_from(["x.py", sug.0]).cmd;
            if let Some(stage) = sug.1 {
                build.config.stage = stage;
            }
            build.build();
        }
    } else {
        println!("help: consider using the `--run` flag to automatically run suggested tests");
    }
}
