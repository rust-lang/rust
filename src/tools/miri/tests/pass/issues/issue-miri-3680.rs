//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

use std::fs::remove_file;
use std::io::{ErrorKind, Seek};

#[path = "../../utils/mod.rs"]
mod utils;

fn main() {
    let path = utils::prepare("miri_test_fs_seek_i64_max_plus_1.txt");

    let mut f = std::fs::File::create(&path).unwrap();
    let error = f.seek(std::io::SeekFrom::Start(i64::MAX as u64 + 1)).unwrap_err();

    // It should be error due to negative offset.
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    // Cleanup
    remove_file(&path).unwrap();
}
