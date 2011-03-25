// xfail-stage0
// error-pattern: Unexpected token 'x'

fn main() {
  let int x = 5;
  alt x;
}
