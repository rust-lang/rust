// -*- rust -*-
use std;
import std._task.yield;

fn main() {
  let int i = 0;
  while (i < 100) {
    i = i + 1;
    log i;
    yield();
  }
}
