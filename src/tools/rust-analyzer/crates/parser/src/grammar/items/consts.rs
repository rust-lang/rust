use super::*;

// test const_item
// const C: u32 = 92;
pub(super) fn konst(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![const]);
    const_or_static(p, m, true);
}

pub(super) fn static_(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![static]);
    const_or_static(p, m, false);
}

fn const_or_static(p: &mut Parser<'_>, m: Marker, is_const: bool) {
    p.eat(T![mut]);

    if is_const && p.eat(T![_]) {
        // test anonymous_const
        // const _: u32 = 0;
    } else {
        // test_err anonymous_static
        // static _: i32 = 5;
        name(p);
    }

    if p.at(T![:]) {
        types::ascription(p);
    } else {
        p.error("missing type for `const` or `static`");
    }
    if p.eat(T![=]) {
        expressions::expr(p);
    }
    p.expect(T![;]);
    m.complete(p, if is_const { CONST } else { STATIC });
}
