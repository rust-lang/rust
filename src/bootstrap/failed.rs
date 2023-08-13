use std::{
    collections::HashSet,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use crate::builder::Builder;

pub const TRACKER_FILE: &'static str = "./src/bootstrap/test.tracker";

/// Runs all the tests that failed on their last run
pub fn failed(_builder: &Builder<'_>) {
    let _failed_tests = get_failed_tests();
}

/// Returns a list of paths of tests that failed the last time that they were run
fn get_failed_tests() -> Vec<PathBuf> {
    let contents = {
        let path = Path::new(TRACKER_FILE);
        let mut f = File::open(path).unwrap_or_else(|_| {
            eprintln!("Please run `x.py test` atleast once before running this command");
            panic!();
        });

        let mut buf = String::new();
        f.read_to_string(&mut buf).expect(&format!("failed to read {}", TRACKER_FILE));

        buf
    };

    let failed_tests =
        serde_json::from_str::<HashSet<&str>>(&contents).expect("failed to deserialise tracker");

    for test in failed_tests {
        let test = test.split(" ").collect::<Vec<_>>();
        println!("{:?}", test);
    }

    vec![]
}
