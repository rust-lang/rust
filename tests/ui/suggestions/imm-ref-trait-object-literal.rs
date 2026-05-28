trait Trait {}

struct S;

impl<'a> Trait for &'a mut S {}

fn foo<X: Trait>(_: X) {}


fn main() {
  let s = S;
  foo(&s); //~ ERROR the trait bound `&S: Trait` is not satisfied
  foo(s); //~ ERROR the trait bound `S: Trait` is not satisfied
}
