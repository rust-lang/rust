// xfail-stage0
// error-pattern:- cannot be applied to type `obj

fn main() {
  auto x = obj(){} - obj(){};
}