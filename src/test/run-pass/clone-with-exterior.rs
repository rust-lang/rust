fn f(@rec(int a, int b) x) {
  check (x.a == 10);
  check (x.b == 12);
}

fn main() {
  let @rec(int a, int b) z = rec(a=10, b=12);
  let task p = spawn thread f(z);
  join p;
}