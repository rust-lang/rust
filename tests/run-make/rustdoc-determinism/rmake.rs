// Assert that the search index is generated deterministically, regardless of the
// order that crates are documented in.

use std::path::Path;

use run_make_support::{diff, rustdoc};

fn main() {
    let foo_first = Path::new("foo_first");
    rustdoc().input("foo.rs").out_dir(&foo_first).run();
    rustdoc().input("bar.rs").out_dir(&foo_first).run();

    let bar_first = Path::new("bar_first");
    rustdoc().input("bar.rs").out_dir(&bar_first).run();
    rustdoc().input("foo.rs").out_dir(&bar_first).run();

    diff()
        .expected_file(foo_first.join("search-index.js"))
        .actual_file(bar_first.join("search-index.js"))
        .run();
}
