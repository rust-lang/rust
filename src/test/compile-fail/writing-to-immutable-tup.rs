// error-pattern: assignment to immutable field
fn main() {
  let tup(int) t = tup(1);
  t._0 = 5;
}
