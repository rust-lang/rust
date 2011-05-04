// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: writing to immutable type
fn main() {
  let vec[int] v = vec(1, 2, 3);
  v.(1) = 4;
}