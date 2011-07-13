// xfail-stage0
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
    fail;
  }
}

fn main() {
  foo(2u);
}
