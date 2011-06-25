// xfail-stage0
// error-pattern:Break outside a loop
fn main() {
  auto pth = break;

  let rec(str t) rs = rec(t=pth);

}