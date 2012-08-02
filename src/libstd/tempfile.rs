//! Temporary files and directories

import core::option;
import option::{none, some};
import rand;
import core::rand::extensions;

fn mkdtemp(prefix: ~str, suffix: ~str) -> option<~str> {
    let r = rand::rng();
    let mut i = 0u;
    while (i < 1000u) {
        let s = prefix + r.gen_str(16u) + suffix;
        if os::make_dir(s, 0x1c0i32) {  // FIXME: u+rwx (#2349)
            return some(s);
        }
        i += 1u;
    }
    return none;
}

#[test]
fn test_mkdtemp() {
    let r = mkdtemp(~"./", ~"foobar");
    alt r {
        some(p) {
            os::remove_dir(p);
            assert(str::ends_with(p, ~"foobar"));
        }
        _ { assert(false); }
    }
}
