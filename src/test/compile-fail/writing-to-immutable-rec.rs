// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: writing to immutable type
fn main() {
  let rec(int x) r = rec(x=1);
  r.x = 6;
}
