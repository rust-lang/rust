// xfail-stage0
fn foo(int x) -> () {
  if check even(x) { 
      log x;
    }
  else {
    fail;
  }
}

fn main() {
  foo(2);
}