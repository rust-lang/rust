// xfail-stage0
// error-pattern:explicit failure
fn f(int a) {
  log a;
}

fn main() { 
  f(fail);
} 
