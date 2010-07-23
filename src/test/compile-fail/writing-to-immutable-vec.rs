// error-pattern: writing to immutable type
fn main() {
  let vec[int] v = vec(1, 2, 3);
  v.(1) = 4;
}