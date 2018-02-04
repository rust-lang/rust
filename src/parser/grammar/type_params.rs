use super::*;

pub(super) fn list(p: &mut Parser) {
    if !p.at(L_ANGLE) {
        return;
    }
    let m = p.start();
    p.bump();

    while !p.at(EOF) && !p.at(R_ANGLE) {
        match p.current() {
            LIFETIME => lifetime_param(p),
            IDENT => type_param(p),
            _ => p.err_and_bump("expected type parameter"),
        }
        if !p.at(R_ANGLE) && !p.expect(COMMA) {
            break;
        }
    }
    p.expect(R_ANGLE);
    m.complete(p, TYPE_PARAM_LIST);

    fn lifetime_param(p: &mut Parser) {
        assert!(p.at(LIFETIME));
        let m = p.start();
        p.bump();
        if p.eat(COLON) {
            while p.at(LIFETIME) {
                p.bump();
                if !p.eat(PLUS) {
                    break;
                }
            }
        }
        m.complete(p, LIFETIME_PARAM);
    }

    fn type_param(p: &mut Parser) {
        assert!(p.at(IDENT));
        let m = p.start();
        p.bump();
        if p.eat(COLON) {
            loop {
                let has_paren = p.eat(L_PAREN);
                p.eat(QUESTION);
                if p.at(FOR_KW) {
                    //TODO
                }
                if p.at(LIFETIME) {
                    p.bump();
                } else if paths::is_path_start(p) {
                    paths::type_path(p);
                } else {
                    break;
                }
                if has_paren {
                    p.expect(R_PAREN);
                }
                if !p.eat(PLUS) {
                    break;
                }
            }
        }
        if p.at(EQ) {
            types::type_ref(p)
        }
        m.complete(p, TYPE_PARAM);
    }
}

pub(super) fn where_clause(p: &mut Parser) {
    if p.at(WHERE_KW) {
        p.bump();
    }
}
