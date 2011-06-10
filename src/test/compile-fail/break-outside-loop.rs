// error-pattern:Break outside a loop
fn main() {
  auto pth = break;

  let rec(str t) res = rec(t=pth);

}