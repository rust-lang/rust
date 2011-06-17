// xfail-stage0
// error-pattern:Number is odd
pred even(uint x) -> bool {
  if (x < 2u) {
    ret false;
  }
  else if (x == 2u) {
    ret true;
  }
  else {
    ret even(x - 2u);
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
