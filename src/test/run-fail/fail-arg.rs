// xfail-stage0
// error-pattern:woe
fn f(int a) {
  log a;
}

fn main() { 
  f(fail "woe");
} 
