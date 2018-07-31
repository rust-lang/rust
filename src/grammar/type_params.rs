use super::*;

pub(super) fn type_param_list(p: &mut Parser) {
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
        name(p);
        if p.at(COLON) {
            bounds(p);
        }
        // test type_param_default
        // struct S<T = i32>;
        if p.at(EQ) {
            p.bump();
            types::type_(p)
        }
        m.complete(p, TYPE_PARAM);
    }
}

// test type_param_bounds
// struct S<T: 'a + ?Sized + (Copy)>;
pub(super) fn bounds(p: &mut Parser) {
    assert!(p.at(COLON));
    p.bump();
    bounds_without_colon(p);
}

pub(super) fn bounds_without_colon(p: &mut Parser) {
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

pub(super) fn where_clause(p: &mut Parser) {
    if p.at(WHERE_KW) {
        let m = p.start();
        p.bump();
        p.expect(IDENT);
        p.expect(COLON);
        p.expect(IDENT);
        m.complete(p, WHERE_CLAUSE);
    }
}
