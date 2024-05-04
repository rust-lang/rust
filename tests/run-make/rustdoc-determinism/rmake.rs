use run_make_support::{diff, rustc, rustdoc, tmp_dir};

/// Assert that the search index is generated deterministically, regardless of the
/// order that crates are documented in.
fn main() {
    let dir_first = tmp_dir().join("first");
    rustdoc().out_dir(&dir_first).input("foo.rs").run();
    rustdoc().out_dir(&dir_first).input("bar.rs").run();

    let dir_second = tmp_dir().join("second");
    rustdoc().out_dir(&dir_second).input("bar.rs").run();
    rustdoc().out_dir(&dir_second).input("foo.rs").run();

    diff()
        .expected_file(dir_first.join("search-index.js"))
        .actual_file(dir_second.join("search-index.js"))
        .run();
}
