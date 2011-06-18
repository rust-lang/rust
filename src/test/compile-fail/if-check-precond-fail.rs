// xfail-stage0
// error-pattern:Unsatisfied precondition constraint
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

fn print_even(uint x) : even(x) {
  log x;
}

fn foo(uint x) -> () {
  if check(even(x)) { 
      fail;
    }
  else {
    print_even(x);
  }
}

fn main() {
  foo(3u);
}
