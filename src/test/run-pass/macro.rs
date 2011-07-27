// xfail-stage0

fn main() {
  #macro([#m1(a), a*4]);
  assert (#m1(2) == 8);
}
