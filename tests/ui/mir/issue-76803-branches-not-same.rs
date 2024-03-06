//@ run-pass

#[derive(Debug, Eq, PartialEq)]
pub enum Type {
    A,
    B,
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
