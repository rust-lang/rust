use std::{
    collections::HashSet,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;

use crate::builder::Builder;
/// Runs all the tests that failed on their last run
pub fn failed(builder: &Builder<'_>) {
    match get_failed_tests() {
        Some(paths) if !paths.is_empty() => {
            let mut build: crate::Build = builder.build.clone();
            build.config.paths = paths;
            build.config.cmd =
                crate::flags::Flags::parse_from(["x.py", "test", "--force-rerun"]).cmd;

            build.build();
        }
        Some(_) => {
            println!("No tests failed on the previous iteration!")
        }
        None => {
            eprintln!("Please run `x.py test` atleast once before running this command.");
            return;
        }
    }
}

/// Returns a list of paths of tests that failed the last time that they were run
fn get_failed_tests() -> Option<Vec<PathBuf>> {
    let contents = {
        let path = get_tracker_file_path();
        let mut f = File::open(path).ok()?;

        let mut buf = String::new();
        f.read_to_string(&mut buf)
            .expect(&format!("failed to read tracker file, a bug report would be appreciated!"));

        buf
    };

    let failed_tests =
        serde_json::from_str::<HashSet<&str>>(&contents).expect("failed to deserialise tracker");

    Some(
        failed_tests
            .iter()
            .map(|test| {
                test.split(" ")
                .filter(|x| {
                    let mut chars = x.chars();
                    !matches!(chars.nth(0).unwrap(), '[' if matches!(chars.last().unwrap(), ']'))
                })
                .collect::<PathBuf>()
            })
            .collect::<Vec<_>>(),
    )
}

pub fn get_tracker_file_path() -> PathBuf {
    let toml = Command::new("cargo")
        .args(["locate-project", "--message-format=plain"])
        .output()
        .expect("failed to locate project root")
        .stdout;

    let root =
        Path::new(std::str::from_utf8(&toml).unwrap().trim()).parent().unwrap().to_path_buf();

    root.join("src/bootstrap/test.tracker")
}
