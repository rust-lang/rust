use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Returns the submodule paths from the `.gitmodules` file in the given directory.
pub fn parse_gitmodules(target_dir: &Path) -> Vec<String> {
    let gitmodules = target_dir.join(".gitmodules");
    assert!(gitmodules.exists(), "'{}' file is missing.", gitmodules.display());

    let file = File::open(gitmodules).unwrap();

    let mut submodules_paths = vec![];
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.starts_with("path") {
            let actual_path = line.split(' ').next_back().expect("Couldn't get value of path");
            submodules_paths.push(actual_path.to_owned());
        }
    }

    submodules_paths
}
