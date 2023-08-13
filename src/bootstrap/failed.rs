use std::{
    collections::HashSet,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use crate::builder::Builder;

use clap::Parser;

pub const TRACKER_FILE: &'static str = "./src/bootstrap/test.tracker";

/// Runs all the tests that failed on their last run
pub fn failed(builder: &Builder<'_>) {
    let mut build: crate::Build = builder.build.clone();
    let failed_tests = get_failed_tests();

    if failed_tests.is_empty() {
        println!("No tests failed on the previous iteration!");
        return;
    }

    build.config.paths = failed_tests;
    build.config.cmd = crate::flags::Flags::parse_from(["x.py", "test", "--force-rerun"]).cmd;

    build.build();
}

/// Returns a list of paths of tests that failed the last time that they were run
fn get_failed_tests() -> Vec<PathBuf> {
    let contents = {
        let path = Path::new(TRACKER_FILE);
        let mut f = File::open(path).unwrap_or_else(|_| {
            eprintln!("Please run `x.py test` atleast once before running this command.");
            panic!();
        });

        let mut buf = String::new();
        f.read_to_string(&mut buf).expect(&format!("failed to read {}", TRACKER_FILE));

        buf
    };

    let failed_tests =
        serde_json::from_str::<HashSet<&str>>(&contents).expect("failed to deserialise tracker");

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
        .collect::<Vec<_>>()
}
