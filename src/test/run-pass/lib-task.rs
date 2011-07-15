

// xfail-stage0

use std;
import std::task;

fn test_sleep() { task::sleep(1000000u); }

fn test_unsupervise() {
  fn f() {
    task::unsupervise();
    fail;
  }
  spawn f();
}

fn main() {
  // FIXME: Why aren't we running this?
  //test_sleep();
  test_unsupervise();
}