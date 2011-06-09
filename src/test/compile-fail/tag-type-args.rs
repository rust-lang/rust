// xfail-stage0
// error-pattern: Wrong number of type arguments

tag quux[T] {
}

fn foo(quux c) -> () {
  assert false;
}

fn main() {
  fail;
}