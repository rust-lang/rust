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
    p.expect(IDENT);
    p.expect(COLON);
    types::type_ref(p);
    p.expect(EQ);
    expressions::expr(p);
    p.expect(SEMI);
}
