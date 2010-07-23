// error-pattern: writing to immutable type
fn main() {
  let tup(int) t = tup(1);
  t._0 = 5;
}
