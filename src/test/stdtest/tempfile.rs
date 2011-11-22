use std;
import std::fs;
import std::option::some;
import std::str;
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
