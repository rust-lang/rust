use super::*;

pub(super) fn literal(p: &mut Parser) -> bool {
    p.eat(INT_NUMBER) || p.eat(FLOAT_NUMBER)
}