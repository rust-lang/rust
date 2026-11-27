use super::{get_path_canonical, get_path_fallback};
use crate::env;
use crate::fs::{File, canonicalize};
use crate::os::windows::io::AsRawHandle;
use crate::test_helpers::tmpdir;

#[test]
/// Test that `get_path_canonical` and `get_path_fallback` return the exact same path.
fn canonicalize_fallback() {
    let t = tmpdir();
    let fname = t.join("hello.txt");
    // This test may break if run in an environment that requires the fallback.
    // So skip it if not in CI.
    if env::var_os("CI").is_none() && canonicalize(&fname).is_err() {
        return;
    }
    let f = File::create(fname).unwrap();
    let canonical = get_path_canonical(f.as_raw_handle()).unwrap();
    let fallback = get_path_fallback(f.as_raw_handle()).unwrap();
    assert_eq!(canonical, fallback);
}
