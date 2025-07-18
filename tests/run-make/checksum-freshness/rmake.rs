//@ needs-target-std
use run_make_support::{rfs, rustc};

fn main() {
    rustc().input("lib.rs").arg("-Zchecksum-hash-algorithm=blake3").emit("dep-info").run();
    let make_file_contents = rfs::read_to_string("lib.d");
    let expected_contents = rfs::read_to_string("expected.d");
    assert_eq!(make_file_contents, expected_contents);
    assert!(!expected_contents.is_empty());
}
