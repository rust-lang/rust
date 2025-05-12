//@ check-fail

#[derive(PartialEq, Eq)]
pub enum Value {
    Boolean(Option<bool>),
    Float(Option<f64>), //~ ERROR the trait bound `f64: Eq` is not satisfied
}

fn main() {
    let a = Value::Float(Some(f64::NAN));
    assert!(a == a);
}
