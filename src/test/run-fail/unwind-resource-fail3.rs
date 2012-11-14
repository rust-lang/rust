// error-pattern:quux
// xfail-test

class faily_box {
  let i: @int;
  new(i: @int) { self.i = i; }
  // What happens to the box pointer owned by this class?
}

impl faily_box : Drop {
    fn finalize() {
        fail "quux";
    }
}

fn main() {
    faily_box(@10);
}
