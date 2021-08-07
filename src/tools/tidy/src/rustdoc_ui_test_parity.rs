//! Tidy check to ensure that `src/test/ui/rustdoc` tests are all present in `src/test/rustdoc-ui`.

use std::collections::HashSet;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    let rustdoc_ui_folder = "src/test/rustdoc-ui";
    let ui_rustdoc_folder = "src/test/ui/rustdoc";
    let mut rustdoc_ui_tests = HashSet::new();
    super::walk_no_read(&path.join("test/rustdoc-ui"), &mut |_| false, &mut |entry| {
        let file_path = entry.path();
        if let Some(ext) = file_path.extension() {
            if ext == "rs" {
                let testname = file_path.file_name().unwrap().to_str().unwrap().to_owned();
                rustdoc_ui_tests.insert(testname);
            }
        }
    });
    super::walk_no_read(&path.join("test/ui/rustdoc"), &mut |_| false, &mut |entry| {
        let file_path = entry.path();
        if let Some(ext) = file_path.extension() {
            if ext == "rs" {
                let testname = file_path.file_name().unwrap().to_str().unwrap().to_owned();
                if !rustdoc_ui_tests.contains(&testname) {
                    tidy_error!(
                        bad,
                        "{}",
                        &format!(
                            "`{}/{}` is missing from `{}`",
                            ui_rustdoc_folder, testname, rustdoc_ui_folder,
                        ),
                    );
                }
            }
        }
    });
}
