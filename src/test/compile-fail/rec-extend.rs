// xfail-stage0
// error-pattern:expected int but found bool

fn main() {

  auto a = rec(foo = 0);

  auto b = rec(foo = true with a);
}
