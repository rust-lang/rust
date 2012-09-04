//! Temporary files and directories

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use core::option;
use option::{None, Some};

fn mkdtemp(tmpdir: &Path, suffix: &str) -> Option<Path> {
    let r = rand::Rng();
    let mut i = 0u;
    while (i < 1000u) {
        let p = tmpdir.push(r.gen_str(16u) +
                            str::from_slice(suffix));
        if os::make_dir(&p, 0x1c0i32) {  // FIXME: u+rwx (#2349)
            return Some(p);
        }
        i += 1u;
    }
    return None;
}

#[test]
fn test_mkdtemp() {
    let r = mkdtemp(&Path("."), "foobar");
    match r {
        Some(p) => {
            os::remove_dir(&p);
            assert(str::ends_with(p.to_str(), "foobar"));
        }
        _ => assert(false)
    }
}
