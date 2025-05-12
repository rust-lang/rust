use crate::grammar::attributes::ATTRIBUTE_FIRST;

use super::*;

// test struct_item
// struct S {}
pub(super) fn strukt(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![struct]);
    struct_or_union(p, m, true);
}

// test union_item
// struct U { i: i32, f: f32 }
pub(super) fn union(p: &mut Parser<'_>, m: Marker) {
    assert!(p.at_contextual_kw(T![union]));
    p.bump_remap(T![union]);
    struct_or_union(p, m, false);
}

fn struct_or_union(p: &mut Parser<'_>, m: Marker, is_struct: bool) {
    name_r(p, ITEM_RECOVERY_SET);
    generic_params::opt_generic_param_list(p);
    match p.current() {
        T![where] => {
            generic_params::opt_where_clause(p);
            match p.current() {
                T![;] => p.bump(T![;]),
                T!['{'] => record_field_list(p),
                _ => {
                    //FIXME: special case `(` error message
                    p.error("expected `;` or `{`");
                }
            }
        }
        T!['{'] => record_field_list(p),
        // test unit_struct
        // struct S;
        T![;] if is_struct => {
            p.bump(T![;]);
        }
        // test tuple_struct
        // struct S(String, usize);
        T!['('] if is_struct => {
            tuple_field_list(p);
            // test tuple_struct_where
            // struct S<T>(T) where T: Clone;
            generic_params::opt_where_clause(p);
            p.expect(T![;]);
        }
        _ => p.error(if is_struct { "expected `;`, `{`, or `(`" } else { "expected `{`" }),
    }
    m.complete(p, if is_struct { STRUCT } else { UNION });
}

pub(super) fn enum_(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![enum]);
    name_r(p, ITEM_RECOVERY_SET);
    generic_params::opt_generic_param_list(p);
    generic_params::opt_where_clause(p);
    if p.at(T!['{']) {
        variant_list(p);
    } else {
        p.error("expected `{`");
    }
    m.complete(p, ENUM);
}

pub(crate) fn variant_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(EOF) && !p.at(T!['}']) {
        if p.at(T!['{']) {
            error_block(p, "expected enum variant");
            continue;
        }
        variant(p);
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, VARIANT_LIST);

    fn variant(p: &mut Parser<'_>) {
        let m = p.start();
        attributes::outer_attrs(p);
        if p.at(IDENT) {
            name(p);
            match p.current() {
                T!['{'] => record_field_list(p),
                T!['('] => tuple_field_list(p),
                _ => (),
            }

            // test variant_discriminant
            // enum E { X(i32) = 10 }
            if p.eat(T![=]) {
                expressions::expr(p);
            }
            m.complete(p, VARIANT);
        } else {
            m.abandon(p);
            p.err_and_bump("expected enum variant");
        }
    }
}

// test record_field_list
// struct S { a: i32, b: f32, unsafe c: u8 }
pub(crate) fn record_field_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(T!['}']) && !p.at(EOF) {
        if p.at(T!['{']) {
            error_block(p, "expected field");
            continue;
        }
        record_field(p);
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, RECORD_FIELD_LIST);

    fn record_field(p: &mut Parser<'_>) {
        let m = p.start();
        // test record_field_attrs
        // struct S { #[attr] f: f32 }
        attributes::outer_attrs(p);
        opt_visibility(p, false);
        p.eat(T![unsafe]);
        if p.at(IDENT) {
            name(p);
            p.expect(T![:]);
            types::type_(p);
            // test record_field_default_values
            // struct S { f: f32 = 0.0 }
            if p.eat(T![=]) {
                expressions::expr(p);
            }
            m.complete(p, RECORD_FIELD);
        } else {
            m.abandon(p);
            p.err_and_bump("expected field declaration");
        }
    }
}

const TUPLE_FIELD_FIRST: TokenSet =
    types::TYPE_FIRST.union(ATTRIBUTE_FIRST).union(VISIBILITY_FIRST);

// test_err tuple_field_list_recovery
// struct S(struct S;
// struct S(A,,B);
fn tuple_field_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['(']));
    let m = p.start();
    delimited(
        p,
        T!['('],
        T![')'],
        T![,],
        || "expected tuple field".into(),
        TUPLE_FIELD_FIRST,
        |p| {
            let m = p.start();
            // test tuple_field_attrs
            // struct S (#[attr] f32);
            attributes::outer_attrs(p);
            let has_vis = opt_visibility(p, true);
            if !p.at_ts(types::TYPE_FIRST) {
                p.error("expected a type");
                if has_vis {
                    m.complete(p, ERROR);
                } else {
                    m.abandon(p);
                }
                return false;
            }
            types::type_(p);
            m.complete(p, TUPLE_FIELD);
            true
        },
    );

    m.complete(p, TUPLE_FIELD_LIST);
}
