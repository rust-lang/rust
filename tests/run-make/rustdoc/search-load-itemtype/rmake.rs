//@ ignore-cross-compile
//@ needs-crate-type: proc-macro

// Test that rustdoc can deserialize a search index with every itemtype.
// https://github.com/rust-lang/rust/pull/146117

use std::path::Path;

use run_make_support::{htmldocck, rfs, rustdoc, source_root};

fn main() {
    let out_dir = Path::new("rustdoc-search-load-itemtype");

    rfs::create_dir_all(&out_dir);
    rustdoc().out_dir(&out_dir).input("foo.rs").run();
    rustdoc().out_dir(&out_dir).input("bar.rs").arg("--crate-type=proc-macro").run();
    rustdoc().out_dir(&out_dir).input("baz.rs").run();
    htmldocck().arg(out_dir).arg("foo.rs").run();
    htmldocck().arg(out_dir).arg("bar.rs").run();
    htmldocck().arg(out_dir).arg("baz.rs").run();
}
