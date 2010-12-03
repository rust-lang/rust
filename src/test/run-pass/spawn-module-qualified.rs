use std;
import std._task.join;

fn main() {
  auto x = spawn m.child(10);
  join(x);
}
mod m {
  fn child(int i) {
    log i;
  }
}
