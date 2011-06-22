// error-pattern: assignment to immutable field
fn main() {
  let rec(int x) r = rec(x=1);
  r.x = 6;
}
