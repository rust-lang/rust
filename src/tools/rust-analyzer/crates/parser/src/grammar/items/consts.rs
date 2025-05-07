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

    // FIXME: Recover on statics with generic params/where clause.
    if is_const {
        // test generic_const
        // const C<i32>: u32 = 0;
        // impl Foo {
        //     const C<'a>: &'a () = &();
        // }
        generic_params::opt_generic_param_list(p);
    }
    // test_err generic_static
    // static C<i32>: u32 = 0;

    if p.at(T![:]) {
        types::ascription(p);
    } else {
        p.error("missing type for `const` or `static`");
    }
    if p.eat(T![=]) {
        expressions::expr(p);
    }

    if is_const {
        // test const_where_clause
        // const C<i32>: u32 = 0
        // where i32: Copy;
        // trait Foo {
        //     const C: i32 where i32: Copy;
        // }
        generic_params::opt_where_clause(p);
    }
    // test_err static_where_clause
    // static C: u32 = 0
    // where i32: Copy;

    p.expect(T![;]);
    m.complete(p, if is_const { CONST } else { STATIC });
}
