// xfail-stage0
// error-pattern: nonexistent module
import vec;

fn main() {
  auto foo = vec.len([]);
}