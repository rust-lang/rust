// Assert that the search index is generated deterministically, regardless of the
// order that crates are documented in.

//@ needs-target-std

use run_make_support::{diff, path, rustdoc};

fn main() {
    let foo_first = path("foo_first");
    rustdoc().input("foo.rs").out_dir(&foo_first).run();
    rustdoc().input("bar.rs").out_dir(&foo_first).run();

    let bar_first = path("bar_first");
    rustdoc().input("bar.rs").out_dir(&bar_first).run();
    rustdoc().input("foo.rs").out_dir(&bar_first).run();

    diff()
        .expected_file(foo_first.join("search-index.js"))
        .actual_file(bar_first.join("search-index.js"))
        .run();
}
