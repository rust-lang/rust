// error-pattern:fail
// xfail-test

class r {
  new(i:int) {}
  drop { fail; }
}

fn main() {
    @0;
    let r <- r(0);
}