struct Foo;

fn main() {
  let mut a = Foo;
  let ref b = Foo;
  a += *b; //~ Error: binary assignment operation `+=` cannot be applied to type `Foo`
}
