use super::*;

pub(super) fn opt_type_param_list(p: &mut Parser) {
    if !p.at(L_ANGLE) {
        return;
    }
    type_param_list(p);
}

fn type_param_list(p: &mut Parser) {
    assert!(p.at(L_ANGLE));
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
}

fn lifetime_param(p: &mut Parser) {
    assert!(p.at(LIFETIME));
    let m = p.start();
    p.bump();
    if p.at(COLON) {
        lifetime_bounds(p);
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

// test type_param_bounds
// struct S<T: 'a + ?Sized + (Copy)>;
pub(super) fn bounds(p: &mut Parser) {
    assert!(p.at(COLON));
    p.bump();
    bounds_without_colon(p);
}

fn lifetime_bounds(p: &mut Parser) {
    assert!(p.at(COLON));
    p.bump();
    while p.at(LIFETIME) {
        p.bump();
        if !p.eat(PLUS) {
            break;
        }
    }
}

pub(super) fn bounds_without_colon(p: &mut Parser) {
    loop {
        let has_paren = p.eat(L_PAREN);
        p.eat(QUESTION);
        match p.current() {
            LIFETIME => p.bump(),
            FOR_KW => types::for_type(p),
            _ if paths::is_path_start(p) => types::path_type(p),
            _ => break,
        }
        if has_paren {
            p.expect(R_PAREN);
        }
        if !p.eat(PLUS) {
            break;
        }
    }
}

// test where_clause
// fn foo()
// where
//    'a: 'b + 'c,
//    T: Clone + Copy + 'static,
//    Iterator::Item: 'a,
//    <T as Iterator>::Item: 'a
// {}
pub(super) fn opt_where_clause(p: &mut Parser) {
    if !p.at(WHERE_KW) {
        return;
    }
    let m = p.start();
    p.bump();

    if is_where_clause_end(p) {
        // Empty where clause
    } else {
        loop {
            where_predicate(p);

            let comma = p.eat(COMMA);

            if is_where_clause_end(p) {
                break;
            }

            if !comma {
                p.error("expected comma")
            }
        }
    }

    m.complete(p, WHERE_CLAUSE);
}

fn is_where_clause_end(p: &mut Parser) -> bool {
    p.current() == L_CURLY || p.current() == SEMI || p.current() == EQ
}

fn where_predicate(p: &mut Parser) {
    let m = p.start();
    match p.current() {
        LIFETIME => {
            p.bump();
            if p.at(COLON) {
                lifetime_bounds(p);
            } else {
                p.error("expected colon");
            }
        }
        IMPL_KW => {
            p.error("expected lifetime or type");
            return;
        }
        _ => {
            types::type_(p);

            if p.at(COLON) {
                bounds(p);
            } else {
                p.error("expected colon");
            }
        }
    }
    m.complete(p, WHERE_PRED);
}
