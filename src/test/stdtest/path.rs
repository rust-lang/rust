import core::*;

// Testing a few of the path manipuation functions

use std;

import std::fs;
import std::os;

#[test]
fn test() {
    assert (!fs::path_is_absolute("test-path"));

    log(debug, "Current working directory: " + os::getcwd());

    log(debug, fs::make_absolute("test-path"));
    log(debug, fs::make_absolute("/usr/bin"));
}
