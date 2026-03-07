//@ run-pass
//@ reference: expr.arith-logic.syntax
//@ reference: patterns.literal.syntax
//@ reference: patterns.literal.intro
// Check that issue #954 stays fixed


pub fn main() {
    match -1 { -1 => {}, _ => panic!("wat") }
    assert_eq!(1-1, 0);
}
