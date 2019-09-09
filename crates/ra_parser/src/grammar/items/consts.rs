use super::*;

pub(super) fn static_def(p: &mut Parser, m: Marker) {
    const_or_static(p, m, T![static], STATIC_DEF)
}

pub(super) fn const_def(p: &mut Parser, m: Marker) {
    const_or_static(p, m, T![const], CONST_DEF)
}

fn const_or_static(p: &mut Parser, m: Marker, kw: SyntaxKind, def: SyntaxKind) {
    assert!(p.at(kw));
    p.bump_any();
    p.eat(T![mut]); // FIXME: validator to forbid const mut
    name(p);
    types::ascription(p);
    if p.eat(T![=]) {
        expressions::expr(p);
    }
    p.expect(T![;]);
    m.complete(p, def);
}
