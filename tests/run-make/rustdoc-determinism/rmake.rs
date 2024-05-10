// Assert that the search index is generated deterministically, regardless of the
// order that crates are documented in.

use run_make_support::{diff, rustdoc, tmp_dir};

fn main() {
    let foo_first = tmp_dir().join("foo_first");
    rustdoc().input("foo.rs").output(&foo_first).run();
    rustdoc().input("bar.rs").output(&foo_first).run();

    let bar_first = tmp_dir().join("bar_first");
    rustdoc().input("bar.rs").output(&bar_first).run();
    rustdoc().input("foo.rs").output(&bar_first).run();

    diff()
        .expected_file(foo_first.join("search-index.js"))
        .actual_file(bar_first.join("search-index.js"))
        .run();
}
