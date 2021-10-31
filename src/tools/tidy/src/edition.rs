//! Tidy check to ensure that crate `edition` is '2018' or '2021'.

use std::path::Path;

fn is_edition_2018(mut line: &str) -> bool {
    line = line.trim();
    line == "edition = \"2018\""
}

fn is_edition_2021(mut line: &str) -> bool {
    line = line.trim();
    line == "edition = \"2021\""
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

            // Library crates are not yet ready to migrate to 2021.
            if path.components().any(|c| c.as_os_str() == "library") {
                let has = contents.lines().any(is_edition_2018);
                if !has {
                    tidy_error!(
                        bad,
                        "{} doesn't have `edition = \"2018\"` on a separate line",
                        file.display()
                    );
                }
            } else {
                let is_2021 = contents.lines().any(is_edition_2021);
                if !is_2021 {
                    tidy_error!(
                        bad,
                        "{} doesn't have `edition = \"2021\"` on a separate line",
                        file.display()
                    );
                }
            }
        },
    );
}
