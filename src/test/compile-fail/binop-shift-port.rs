// xfail-stage0
// error-pattern:>> cannot be applied to type `port\[int\]`

fn main() {
  let port[int] p1 = port();
  let port[int] p2 = port();
  auto x = p1 >> p2;
}