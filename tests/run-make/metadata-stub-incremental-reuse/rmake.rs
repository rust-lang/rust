//@ needs-target-std
//
// Under `-Zembed-metadata=no` the rlib carries only a metadata stub, which
// rustc writes as a separate file alongside the full metadata. Recompiling
// with `-C incremental` can reuse the full metadata from a work product, and
// the stub must still be produced on that path.

use run_make_support::{rfs, rustc};

fn build(out_dir: &str, incremental: &str) {
    rustc()
        .input("foo.rs")
        .crate_name("foo")
        .crate_type("lib")
        .emit("dep-info,metadata,link")
        .arg("-Zembed-metadata=no")
        .incremental(incremental)
        .out_dir(out_dir)
        .run();
}

fn main() {
    rfs::create_dir("out");
    rfs::create_dir("incr");

    // Populates the incremental session directory, including the `metadata`
    // work product.
    build("out", "incr");

    // Recompiling from unchanged sources marks the metadata dep-node green and
    // reuses that work product.
    build("out", "incr");
}
