//@ check-fail

#[derive(PartialEq, Eq)]
pub enum Value {
    Boolean(Option<bool>),
    Float(Option<f64>), //~ ERROR trait `Eq` is not implemented for `f64`
}

fn main() {
    let a = Value::Float(Some(f64::NAN));
    assert!(a == a);
}
