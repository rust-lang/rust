// Just verify that we successfully run and produce dep graphs when requested.

//@ ignore-cross-compile

use run_make_support::{path, rustc};

fn main() {
    rustc()
        .input("foo.rs")
        .incremental(path("incr"))
        .arg("-Zquery-dep-graph")
        .arg("-Zdump-dep-graph")
        .env("RUST_DEP_GRAPH", path("dep-graph"))
        .run();

    assert!(path("dep-graph.txt").is_file());
    assert!(path("dep-graph.dot").is_file());
}
