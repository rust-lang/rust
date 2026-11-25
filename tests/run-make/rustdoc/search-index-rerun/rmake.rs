// This test ensures that if nothing changed between two rustdoc runs, it will
// still generate the same `search.index` files.
// Regression test for <https://github.com/rust-lang/rust/issues/166236>.

//@ needs-target-std

use std::collections::HashMap;

use run_make_support::path_helpers::read_dir_entries_recursive;
use run_make_support::rfs::read_to_string;
use run_make_support::{diff, rustdoc};

fn index_files() -> HashMap<String, String> {
    rustdoc().input("foo.rs").edition("2015").run();
    let mut files = HashMap::new();
    read_dir_entries_recursive("doc/search.index", |entry| {
        if entry.is_file() {
            let path = entry.to_path_buf();
            let content = read_to_string(&path);
            files.insert(path.display().to_string(), content);
        }
    });
    assert!(!files.is_empty());
    files
}

fn main() {
    let expected_files = index_files();

    for _ in 0..4 {
        let new_files = index_files();
        assert_eq!(expected_files.len(), new_files.len());

        for (file, content) in &expected_files {
            let Some(new_content) = new_files.get(file) else {
                panic!("Missing {file:?} in new content");
            };
            diff().expected_text(file, content).actual_text(file, new_content).run();
        }
    }
}
