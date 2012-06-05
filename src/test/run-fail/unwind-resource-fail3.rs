// error-pattern:quux
// xfail-test

class faily_box {
  let i: @int;
  new(i: @int) { self.i = i; }
  // What happens to the box pointer owned by this class?
  drop { fail "quux"; }
}

fn main() {
    faily_box(@10);
}
