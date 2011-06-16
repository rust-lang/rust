// xfail-stage0
// error-pattern:Number is odd
pred even(uint x) -> bool {
  if (x < 2) {
    ret false;
  }
  else if (x == 2) {
    ret true;
  }
  else {
    ret even(x - 2);
  }
}

fn foo(uint x) -> () {
  if check(even(x)) { 
      log x;
    }
  else {
    fail "Number is odd";
  }
}

fn main() {
  foo(3u);
}
