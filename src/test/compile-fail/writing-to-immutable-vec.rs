// xfail-stage0
// error-pattern:assignment to immutable vec content
fn main() {
  let vec[int] v = [1, 2, 3];
  v.(1) = 4;
}
