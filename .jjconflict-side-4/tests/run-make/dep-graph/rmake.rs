// Just verify that we successfully run and produce dep graphs when requested. The parallel
// frontend is covered too, because it leaves unused dep-graph indices that the dump must skip.

//@ ignore-cross-compile

use run_make_support::{path, rustc};

fn main() {
    for threads in ["1", "2"] {
        rustc()
            .input("foo.rs")
            .incremental(path(format!("incr-{threads}")))
            .arg("-Zquery-dep-graph")
            .arg("-Zdump-dep-graph")
            .arg(format!("-Zthreads={threads}"))
            .env("RUST_DEP_GRAPH", path(format!("dep-graph-{threads}")))
            .run();

        assert!(path(format!("dep-graph-{threads}.txt")).is_file());
        assert!(path(format!("dep-graph-{threads}.dot")).is_file());
    }
}
