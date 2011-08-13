use std;
import std::task::join_id;
import std::task::_spawn;

fn main() {
  let x = _spawn(bind m::child(10));
  join_id(x);
}
mod m {
  fn child(i: int) {
    log i;
  }
}
