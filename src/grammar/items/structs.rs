use super::*;

pub(super) fn struct_item(p: &mut Parser) {
    assert!(p.at(STRUCT_KW));
    p.bump();

    name(p);
    type_params::type_param_list(p);
    match p.current() {
        WHERE_KW => {
            type_params::where_clause(p);
            match p.current() {
                SEMI => {
                    p.bump();
                    return;
                }
                L_CURLY => named_fields(p),
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
        L_CURLY => named_fields(p),
        L_PAREN => {
            pos_fields(p);
            p.expect(SEMI);
        }
        _ => {
            p.error("expected `;`, `{`, or `(`");
            return;
        }
    }
}

pub(super) fn enum_item(p: &mut Parser) {
    assert!(p.at(ENUM_KW));
    p.bump();
    name(p);
    type_params::type_param_list(p);
    type_params::where_clause(p);
    if p.expect(L_CURLY) {
        while !p.at(EOF) && !p.at(R_CURLY) {
            let var = p.start();
            attributes::outer_attributes(p);
            if p.at(IDENT) {
                name(p);
                match p.current() {
                    L_CURLY => named_fields(p),
                    L_PAREN => pos_fields(p),
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
    }
}

fn named_fields(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    while !p.at(R_CURLY) && !p.at(EOF) {
        named_field(p);
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);

    fn named_field(p: &mut Parser) {
        let field = p.start();
        visibility(p);
        if p.at(IDENT) {
            name(p);
            p.expect(COLON);
            types::type_(p);
            field.complete(p, NAMED_FIELD);
        } else {
            field.abandon(p);
            p.err_and_bump("expected field declaration");
        }
    }
}

fn pos_fields(p: &mut Parser) {
    if !p.expect(L_PAREN) {
        return;
    }
    while !p.at(R_PAREN) && !p.at(EOF) {
        let pos_field = p.start();
        visibility(p);
        types::type_(p);
        pos_field.complete(p, POS_FIELD);

        if !p.at(R_PAREN) {
            p.expect(COMMA);
        }
    }
    p.expect(R_PAREN);
}
