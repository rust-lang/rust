// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern:explicit failure
fn f(int a) {
  log a;
}

fn main() { 
  f(fail);
} 
