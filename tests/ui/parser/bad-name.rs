//@ error-pattern: expected

fn main() {
  let x.y::<isize>.z foo;
  //^ error: expected a pattern, found an expression
}
