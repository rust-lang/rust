//! FIXME: write short doc here

use super::*;

pub(super) fn struct_def(p: &mut Parser, m: Marker) {
    assert!(p.at(T![struct]));
    p.bump(T![struct]);
    struct_or_union(p, m, T![struct], STRUCT_DEF);
}

pub(super) fn union_def(p: &mut Parser, m: Marker) {
    assert!(p.at_contextual_kw("union"));
    p.bump_remap(T![union]);
    struct_or_union(p, m, T![union], UNION_DEF);
}

fn struct_or_union(p: &mut Parser, m: Marker, kw: SyntaxKind, def: SyntaxKind) {
    name_r(p, ITEM_RECOVERY_SET);
    type_params::opt_type_param_list(p);
    match p.current() {
        T![where] => {
            type_params::opt_where_clause(p);
            match p.current() {
                T![;] => {
                    p.bump(T![;]);
                }
                T!['{'] => record_field_def_list(p),
                _ => {
                    //FIXME: special case `(` error message
                    p.error("expected `;` or `{`");
                }
            }
        }
        T![;] if kw == T![struct] => {
            p.bump(T![;]);
        }
        T!['{'] => record_field_def_list(p),
        T!['('] if kw == T![struct] => {
            tuple_field_def_list(p);
            // test tuple_struct_where
            // struct Test<T>(T) where T: Clone;
            // struct Test<T>(T);
            type_params::opt_where_clause(p);
            p.expect(T![;]);
        }
        _ if kw == T![struct] => {
            p.error("expected `;`, `{`, or `(`");
        }
        _ => {
            p.error("expected `{`");
        }
    }
    m.complete(p, def);
}

pub(super) fn enum_def(p: &mut Parser, m: Marker) {
    assert!(p.at(T![enum]));
    p.bump(T![enum]);
    name_r(p, ITEM_RECOVERY_SET);
    type_params::opt_type_param_list(p);
    type_params::opt_where_clause(p);
    if p.at(T!['{']) {
        enum_variant_list(p);
    } else {
        p.error("expected `{`")
    }
    m.complete(p, ENUM_DEF);
}

pub(crate) fn enum_variant_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(EOF) && !p.at(T!['}']) {
        if p.at(T!['{']) {
            error_block(p, "expected enum variant");
            continue;
        }
        let var = p.start();
        attributes::outer_attributes(p);
        if p.at(IDENT) {
            name(p);
            match p.current() {
                T!['{'] => record_field_def_list(p),
                T!['('] => tuple_field_def_list(p),
                _ => (),
            }

            // test variant_discriminant
            // enum E { X(i32) = 10 }
            if p.eat(T![=]) {
                expressions::expr(p);
            }
            var.complete(p, ENUM_VARIANT);
        } else {
            var.abandon(p);
            p.err_and_bump("expected enum variant");
        }
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, ENUM_VARIANT_LIST);
}

pub(crate) fn record_field_def_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    while !p.at(T!['}']) && !p.at(EOF) {
        if p.at(T!['{']) {
            error_block(p, "expected field");
            continue;
        }
        record_field_def(p);
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, RECORD_FIELD_DEF_LIST);

    fn record_field_def(p: &mut Parser) {
        let m = p.start();
        // test record_field_attrs
        // struct S {
        //     #[serde(with = "url_serde")]
        //     pub uri: Uri,
        // }
        attributes::outer_attributes(p);
        opt_visibility(p);
        if p.at(IDENT) {
            name(p);
            p.expect(T![:]);
            types::type_(p);
            m.complete(p, RECORD_FIELD_DEF);
        } else {
            m.abandon(p);
            p.err_and_bump("expected field declaration");
        }
    }
}

fn tuple_field_def_list(p: &mut Parser) {
    assert!(p.at(T!['(']));
    let m = p.start();
    if !p.expect(T!['(']) {
        return;
    }
    while !p.at(T![')']) && !p.at(EOF) {
        let m = p.start();
        // test tuple_field_attrs
        // struct S (
        //     #[serde(with = "url_serde")]
        //     pub Uri,
        // );
        //
        // enum S {
        //     Uri(#[serde(with = "url_serde")] Uri),
        // }
        attributes::outer_attributes(p);
        opt_visibility(p);
        if !p.at_ts(types::TYPE_FIRST) {
            p.error("expected a type");
            m.complete(p, ERROR);
            break;
        }
        types::type_(p);
        m.complete(p, TUPLE_FIELD_DEF);

        if !p.at(T![')']) {
            p.expect(T![,]);
        }
    }
    p.expect(T![')']);
    m.complete(p, TUPLE_FIELD_DEF_LIST);
}
