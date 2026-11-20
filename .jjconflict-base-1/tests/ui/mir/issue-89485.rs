// Regression test for issue #89485.

//@ run-pass

#[derive(Debug, Eq, PartialEq)]
pub enum Type {
    A = 1,
    B = 2,
}
pub fn encode(v: Type) -> Type {
    match v {
        Type::A => Type::B,
        _ => v,
    }
}
fn main() {
  assert_eq!(Type::B, encode(Type::A));
}
