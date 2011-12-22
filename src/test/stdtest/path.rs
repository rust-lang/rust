import core::*;

// Testing a few of the path manipuation functions

use std;

import std::fs;
import std::os;

#[test]
fn test() {
    assert (!fs::path_is_absolute("test-path"));

    log_full(core::debug, "Current working directory: " + os::getcwd());

    log_full(core::debug, fs::make_absolute("test-path"));
    log_full(core::debug, fs::make_absolute("/usr/bin"));
}
