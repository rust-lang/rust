
// Testing a few of the path manipuation functions

use std;

import std::fs;
import std::os;

#[test]
fn test() {
    assert (!fs::path_is_absolute(~"test-path"));

    log ~"Current working directory: " + os::getcwd();

    log fs::make_absolute(~"test-path");
    log fs::make_absolute(~"/usr/bin");
}
