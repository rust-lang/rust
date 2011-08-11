use std;
import std::task::join;

fn main() {
  let x = spawn m::child(10);
  join(x);
}
mod m {
  fn child(i: int) {
    log i;
  }
}
