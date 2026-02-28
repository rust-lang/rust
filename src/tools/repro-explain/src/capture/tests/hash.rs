use std::fs;

use tempfile::tempdir;

use super::{sha256_file, sha256_files_parallel};

#[test]
fn computes_hashes_in_parallel_for_multiple_files() {
    let dir = tempdir().expect("tempdir");
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    fs::write(&a, "alpha").expect("write a");
    fs::write(&b, "beta").expect("write b");

    let map = sha256_files_parallel(&[a.clone(), b.clone()]).expect("parallel hash");
    assert_eq!(map.get(&a), Some(&sha256_file(&a).expect("hash a")));
    assert_eq!(map.get(&b), Some(&sha256_file(&b).expect("hash b")));
}
