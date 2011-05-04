// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern: Unexpected token 'x'

fn main() {
  let int x = 5;
  alt x;
}
