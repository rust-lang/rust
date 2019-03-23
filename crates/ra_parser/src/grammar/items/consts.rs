use super::*;

pub(super) fn static_def(p: &mut Parser, m: Marker) {
    const_or_static(p, m, STATIC_KW, STATIC_DEF)
}

pub(super) fn const_def(p: &mut Parser, m: Marker) {
    const_or_static(p, m, CONST_KW, CONST_DEF)
}

fn const_or_static(p: &mut Parser, m: Marker, kw: SyntaxKind, def: SyntaxKind) {
    assert!(p.at(kw));
    p.bump();
    p.eat(MUT_KW); // FIXME: validator to forbid const mut
    name(p);
    types::ascription(p);
    if p.eat(EQ) {
        expressions::expr(p);
    }
    p.expect(SEMI);
    m.complete(p, def);
}
