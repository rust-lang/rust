use super::*;

pub(super) fn konst(p: &mut Parser, m: Marker) {
    p.bump(T![const]);
    const_or_static(p, m, true)
}

pub(super) fn static_(p: &mut Parser, m: Marker) {
    p.bump(T![static]);
    const_or_static(p, m, false)
}

fn const_or_static(p: &mut Parser, m: Marker, is_const: bool) {
    p.eat(T![mut]);

    // Allow `_` in place of an identifier in a `const`.
    let is_const_underscore = is_const && p.eat(T![_]);
    if !is_const_underscore {
        name(p);
    }

    // test_err static_underscore
    // static _: i32 = 5;
    if p.at(T![:]) {
        types::ascription(p);
    } else {
        p.error("missing type for `const` or `static`")
    }
    if p.eat(T![=]) {
        expressions::expr(p);
    }
    p.expect(T![;]);
    m.complete(p, if is_const { CONST } else { STATIC });
}
