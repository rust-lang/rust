// tests that changing the content of a comment will not cause a change in the rmeta of a library if
// -Zsplit-rmeta is enabled
use std::hash::{Hash, Hasher};
use std::path::Path;

use run_make_support::{rfs, rustc};

fn main() {
    let before = check_and_hash("before.rs");
    let after = check_and_hash("after.rs");
    dbg!(before, after);
    assert_eq!(before, after);
}

fn check_and_hash<P>(filename: P) -> u64
where
    P: AsRef<Path>,
{
    rfs::rename(filename, "foo.rs");
    rustc().input("foo.rs").emit("metadata").arg("-Zsplit-rmeta").run();
    // hash the output
    let bytes = rfs::read("libfoo.rmeta");
    let mut hasher = std::hash::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
