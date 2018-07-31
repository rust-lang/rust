use super::*;

pub(super) fn static_item(p: &mut Parser) {
    const_or_static(p, STATIC_KW)
}

pub(super) fn const_item(p: &mut Parser) {
    const_or_static(p, CONST_KW)
}

fn const_or_static(p: &mut Parser, kw: SyntaxKind) {
    assert!(p.at(kw));
    p.bump();
    p.eat(MUT_KW); // TODO: validator to forbid const mut
    name(p);
    types::ascription(p);
    p.expect(EQ);
    expressions::expr(p);
    p.expect(SEMI);
}
