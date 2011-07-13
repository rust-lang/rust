use std;
import std::task;

#[test]
#[ignore]
fn test_sleep() { task::sleep(1000000u); }

#[test]
fn test_unsupervise() {
  fn f() {
    task::unsupervise();
    fail;
  }
  spawn f();
}

#[test]
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
