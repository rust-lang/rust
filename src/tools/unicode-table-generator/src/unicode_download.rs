use std::path::Path;
use std::process::{Command, Output};

use crate::UNICODE_DIRECTORY;

static URL_PREFIX: &str = "https://www.unicode.org/Public/UCD/latest/ucd/";

static README: &str = "ReadMe.txt";

static RESOURCES: &[&str] =
    &["DerivedCoreProperties.txt", "PropList.txt", "UnicodeData.txt", "SpecialCasing.txt"];

#[track_caller]
fn fetch(url: &str) -> Output {
    let output = Command::new("curl").arg(URL_PREFIX.to_owned() + url).output().unwrap();
    if !output.status.success() {
        panic!(
            "Failed to run curl to fetch {url}: stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output
}

pub fn fetch_latest() {
    let directory = Path::new(UNICODE_DIRECTORY);
    if directory.exists() {
        eprintln!(
            "Not refetching unicode data, already exists, please delete {directory:?} to regenerate",
        );
        return;
    }
    if let Err(e) = std::fs::create_dir_all(directory) {
        panic!("Failed to create {UNICODE_DIRECTORY:?}: {e}");
    }
    let output = fetch(README);
    let current = std::fs::read_to_string(directory.join(README)).unwrap_or_default();
    if current.as_bytes() != &output.stdout[..] {
        std::fs::write(directory.join(README), output.stdout).unwrap();
    }

    for resource in RESOURCES {
        let output = fetch(resource);
        std::fs::write(directory.join(resource), output.stdout).unwrap();
    }
}
