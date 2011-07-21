// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
fn f(@rec(int a, int b) x) {
  assert (x.a == 10);
  assert (x.b == 12);
}

fn main() {
  let @rec(int a, int b) z = rec(a=10, b=12);
  let task p = spawn thread f(z);
  join p;
}