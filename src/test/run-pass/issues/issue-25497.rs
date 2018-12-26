// run-pass
#[derive(Clone, Debug, PartialEq)]
enum Expression {
    Dummy,
    Add(Box<Expression>),
}

use Expression::*;

fn simplify(exp: Expression) -> Expression {
    match exp {
        Add(n) => *n.clone(),
        _ => Dummy
    }
}

fn main() {
    assert_eq!(simplify(Add(Box::new(Dummy))), Dummy);
}
