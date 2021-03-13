//! Tidy check to ensure that crate `edition` is '2018'

use std::path::Path;

fn is_edition_2018(mut line: &str) -> bool {
    line = line.trim();
    line == "edition = \"2018\"" || line == "edition = \'2018\'"
}

pub fn check(path: &Path, bad: &mut bool) {
    super::walk(
        path,
        &mut |path| super::filter_dirs(path) || path.ends_with("src/test"),
        &mut |entry, contents| {
            let file = entry.path();
            let filename = file.file_name().unwrap();
            if filename != "Cargo.toml" {
                return;
            }
            let has_edition = contents.lines().any(is_edition_2018);
            if !has_edition {
                tidy_error!(
                    bad,
                    "{} doesn't have `edition = \"2018\"` on a separate line",
                    file.display()
                );
            }
        },
    );
}
