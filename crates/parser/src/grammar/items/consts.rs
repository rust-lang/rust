//! FIXME: write short doc here

use super::*;

pub(super) fn static_(p: &mut Parser, m: Marker) {
    const_or_static(p, m, T![static], STATIC)
}

pub(super) fn konst(p: &mut Parser, m: Marker) {
    const_or_static(p, m, T![const], CONST)
}

fn const_or_static(p: &mut Parser, m: Marker, kw: SyntaxKind, def: SyntaxKind) {
    assert!(p.at(kw));
    p.bump(kw);
    p.eat(T![mut]);

    // Allow `_` in place of an identifier in a `const`.
    let is_const_underscore = kw == T![const] && p.eat(T![_]);
    if !is_const_underscore {
        name(p);
    }

    // test_err static_underscore
    // static _: i32 = 5;

    types::ascription(p);
    if p.eat(T![=]) {
        expressions::expr(p);
    }
    p.expect(T![;]);
    m.complete(p, def);
}
