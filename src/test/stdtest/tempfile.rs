import core::*;

use std;
import option;
import std::fs;
import option::some;
import str;
import std::tempfile;

#[test]
fn mkdtemp() {
    let r = tempfile::mkdtemp("./", "foobar");
    alt r {
        some(p) {
            fs::remove_dir(p);
            assert(str::ends_with(p, "foobar"));
        }
        _ { assert(false); }
    }
}
