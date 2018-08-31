use super::*;

pub(super) fn struct_def(p: &mut Parser) {
    assert!(p.at(STRUCT_KW));
    p.bump();

    name_r(p, ITEM_RECOVERY_SET);
    type_params::opt_type_param_list(p);
    match p.current() {
        WHERE_KW => {
            type_params::opt_where_clause(p);
            match p.current() {
                SEMI => {
                    p.bump();
                    return;
                }
                L_CURLY => named_field_def_list(p),
                _ => {
                    //TODO: special case `(` error message
                    p.error("expected `;` or `{`");
                    return;
                }
            }
        }
        SEMI => {
            p.bump();
            return;
        }
        L_CURLY => named_field_def_list(p),
        L_PAREN => {
            pos_field_list(p);
            p.expect(SEMI);
        }
        _ => {
            p.error("expected `;`, `{`, or `(`");
            return;
        }
    }
}

pub(super) fn enum_def(p: &mut Parser) {
    assert!(p.at(ENUM_KW));
    p.bump();
    name_r(p, ITEM_RECOVERY_SET);
    type_params::opt_type_param_list(p);
    type_params::opt_where_clause(p);
    if p.at(L_CURLY) {
        enum_variant_list(p);
    } else {
        p.error("expected `{`")
    }
}

fn enum_variant_list(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        if p.at(L_CURLY) {
            error_block(p, "expected enum variant");
            continue;
        }
        let var = p.start();
        attributes::outer_attributes(p);
        if p.at(IDENT) {
            name(p);
            match p.current() {
                L_CURLY => named_field_def_list(p),
                L_PAREN => pos_field_list(p),
                EQ => {
                    p.bump();
                    expressions::expr(p);
                }
                _ => (),
            }
            var.complete(p, ENUM_VARIANT);
        } else {
            var.abandon(p);
            p.err_and_bump("expected enum variant");
        }
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
    m.complete(p, ENUM_VARIANT_LIST);
}

pub(crate) fn named_field_def_list(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.bump();
    while !p.at(R_CURLY) && !p.at(EOF) {
        named_field_def(p);
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
    m.complete(p, NAMED_FIELD_DEF_LIST);

    fn named_field_def(p: &mut Parser) {
        let m = p.start();
        // test field_attrs
        // struct S {
        //     #[serde(with = "url_serde")]
        //     pub uri: Uri,
        // }
        attributes::outer_attributes(p);
        opt_visibility(p);
        if p.at(IDENT) {
            name(p);
            p.expect(COLON);
            types::type_(p);
            m.complete(p, NAMED_FIELD_DEF);
        } else {
            m.abandon(p);
            p.err_and_bump("expected field declaration");
        }
    }
}

fn pos_field_list(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    let m = p.start();
    if !p.expect(L_PAREN) {
        return;
    }
    while !p.at(R_PAREN) && !p.at(EOF) {
        let pos_field = p.start();
        opt_visibility(p);
        types::type_(p);
        pos_field.complete(p, POS_FIELD);

        if !p.at(R_PAREN) {
            p.expect(COMMA);
        }
    }
    p.expect(R_PAREN);
    m.complete(p, POS_FIELD_LIST);
}
