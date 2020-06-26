use crate::UNICODE_DIRECTORY;
use std::path::Path;
use std::process::Command;

static URL_PREFIX: &str = "https://www.unicode.org/Public/UCD/latest/ucd/";

static README: &str = "ReadMe.txt";

static RESOURCES: &[&str] =
    &["DerivedCoreProperties.txt", "PropList.txt", "UnicodeData.txt", "SpecialCasing.txt"];

pub fn fetch_latest() {
    let directory = Path::new(UNICODE_DIRECTORY);
    if directory.exists() {
        eprintln!(
            "Not refetching unicode data, already exists, please delete {:?} to regenerate",
            directory
        );
        return;
    }
    if let Err(e) = std::fs::create_dir_all(directory) {
        panic!("Failed to create {:?}: {}", UNICODE_DIRECTORY, e);
    }
    let output = Command::new("curl").arg(URL_PREFIX.to_owned() + README).output().unwrap();
    if !output.status.success() {
        panic!(
            "Failed to run curl to fetch readme: stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let current = std::fs::read_to_string(directory.join(README)).unwrap_or_default();
    if current.as_bytes() != &output.stdout[..] {
        std::fs::write(directory.join(README), output.stdout).unwrap();
    }

    for resource in RESOURCES {
        let output = Command::new("curl").arg(URL_PREFIX.to_owned() + resource).output().unwrap();
        if !output.status.success() {
            panic!(
                "Failed to run curl to fetch {}: stderr: {}",
                resource,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::fs::write(directory.join(resource), output.stdout).unwrap();
    }
}
