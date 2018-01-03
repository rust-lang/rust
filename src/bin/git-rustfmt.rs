extern crate env_logger;
extern crate getopts;
#[macro_use]
extern crate log;
extern crate rustfmt_nightly as rustfmt;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use getopts::{Matches, Options};

use rustfmt::{run, Input};
use rustfmt::config;

fn prune_files(files: Vec<&str>) -> Vec<&str> {
    let prefixes: Vec<_> = files
        .iter()
        .filter(|f| f.ends_with("mod.rs") || f.ends_with("lib.rs"))
        .map(|f| &f[..f.len() - 6])
        .collect();

    let mut pruned_prefixes = vec![];
    for p1 in prefixes {
        let mut include = true;
        if !p1.starts_with("src/bin/") {
            for p2 in &pruned_prefixes {
                if p1.starts_with(p2) {
                    include = false;
                    break;
                }
            }
        }
        if include {
            pruned_prefixes.push(p1);
        }
    }
    debug!("prefixes: {:?}", pruned_prefixes);

    files
        .into_iter()
        .filter(|f| {
            let mut include = true;
            if f.ends_with("mod.rs") || f.ends_with("lib.rs") || f.starts_with("src/bin/") {
                return true;
            }
            for pp in &pruned_prefixes {
                if f.starts_with(pp) {
                    include = false;
                    break;
                }
            }
            include
        })
        .collect()
}

fn git_diff(commits: &str) -> String {
    let mut cmd = Command::new("git");
    cmd.arg("diff");
    if commits != "0" {
        cmd.arg(format!("HEAD~{}", commits));
    }
    let output = cmd.output().expect("Couldn't execute `git diff`");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn get_files(input: &str) -> Vec<&str> {
    input
        .lines()
        .filter(|line| line.starts_with("+++ b/") && line.ends_with(".rs"))
        .map(|line| &line[6..])
        .collect()
}

fn fmt_files(files: &[&str]) -> i32 {
    let (config, _) = config::Config::from_resolved_toml_path(Path::new("."))
        .unwrap_or_else(|_| (config::Config::default(), None));

    let mut exit_code = 0;
    for file in files {
        let summary = run(Input::File(PathBuf::from(file)), &config);
        if !summary.has_no_errors() {
            exit_code = 1;
        }
    }
    exit_code
}

fn uncommitted_files() -> Vec<String> {
    let mut cmd = Command::new("git");
    cmd.arg("ls-files");
    cmd.arg("--others");
    cmd.arg("--modified");
    cmd.arg("--exclude-standard");
    let output = cmd.output().expect("Couldn't execute Git");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter(|s| s.ends_with(".rs"))
        .map(|s| s.to_owned())
        .collect()
}

fn check_uncommitted() {
    let uncommitted = uncommitted_files();
    debug!("uncommitted files: {:?}", uncommitted);
    if !uncommitted.is_empty() {
        println!("Found untracked changes:");
        for f in &uncommitted {
            println!("  {}", f);
        }
        println!("Commit your work, or run with `-u`.");
        println!("Exiting.");
        std::process::exit(1);
    }
}

fn make_opts() -> Options {
    let mut opts = Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optflag("c", "check", "check only, don't format (unimplemented)");
    opts.optflag("u", "uncommitted", "format uncommitted files");
    opts
}

struct Config {
    commits: String,
    uncommitted: bool,
    check: bool,
}

impl Config {
    fn from_args(matches: &Matches, opts: &Options) -> Config {
        // `--help` display help message and quit
        if matches.opt_present("h") {
            let message = format!(
                "\nusage: {} <commits> [options]\n\n\
                 commits: number of commits to format, default: 1",
                env::args_os().next().unwrap().to_string_lossy()
            );
            println!("{}", opts.usage(&message));
            std::process::exit(0);
        }

        let mut config = Config {
            commits: "1".to_owned(),
            uncommitted: false,
            check: false,
        };

        if matches.opt_present("c") {
            config.check = true;
            unimplemented!();
        }

        if matches.opt_present("u") {
            config.uncommitted = true;
        }

        if matches.free.len() > 1 {
            panic!("unknown arguments, use `-h` for usage");
        }
        if matches.free.len() == 1 {
            let commits = matches.free[0].trim();
            if u32::from_str(commits).is_err() {
                panic!("Couldn't parse number of commits");
            }
            config.commits = commits.to_owned();
        }

        config
    }
}

fn main() {
    let _ = env_logger::init();

    let opts = make_opts();
    let matches = opts.parse(env::args().skip(1))
        .expect("Couldn't parse command line");
    let config = Config::from_args(&matches, &opts);

    if !config.uncommitted {
        check_uncommitted();
    }

    let stdout = git_diff(&config.commits);
    let files = get_files(&stdout);
    debug!("files: {:?}", files);
    let files = prune_files(files);
    debug!("pruned files: {:?}", files);
    let exit_code = fmt_files(&files);
    std::process::exit(exit_code);
}
