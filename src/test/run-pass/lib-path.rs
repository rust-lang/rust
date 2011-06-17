// xfail-stage0

// Testing a few of the path manipuation functions

use std;

import std::fs;
import std::os;

fn main() {
  assert(!fs::path_is_absolute("test-path"));

  log "Current working directory: " + os::getcwd();

  log fs::make_absolute("test-path");
  log fs::make_absolute("/usr/bin");
}