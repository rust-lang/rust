

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

fn test_join() {
  fn winner() {
  }

  auto wintask = spawn winner();

  assert task::join(wintask) == task::tr_success;

  fn failer() {
    task::unsupervise();
    fail;
  }

  auto failtask = spawn failer();

  assert task::join(failtask) == task::tr_failure;
}

fn main() {
  // FIXME: Why aren't we running this?
  //test_sleep();
  test_unsupervise();
  test_join();
}