// Assert that the search index is generated deterministically, regardless of the
// order that crates are documented in.

use run_make_support::{diff, rustdoc};
use std::path::Path;

fn main() {
    let foo_first = Path::new("foo_first");
    rustdoc().input("foo.rs").output(&foo_first).run();
    rustdoc().input("bar.rs").output(&foo_first).run();

    let bar_first = Path::new("bar_first");
    rustdoc().input("bar.rs").output(&bar_first).run();
    rustdoc().input("foo.rs").output(&bar_first).run();

    diff()
        .expected_file(foo_first.join("search-index.js"))
        .actual_file(bar_first.join("search-index.js"))
        .run();
}
