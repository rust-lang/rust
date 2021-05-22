use super::*;

pub(super) fn opt_generic_arg_list(p: &mut Parser, colon_colon_required: bool) {
    let m;
    if p.at(T![::]) && p.nth(2) == T![<] {
        m = p.start();
        p.bump(T![::]);
        p.bump(T![<]);
    } else if !colon_colon_required && p.at(T![<]) && p.nth(1) != T![=] {
        m = p.start();
        p.bump(T![<]);
    } else {
        return;
    }

    while !p.at(EOF) && !p.at(T![>]) {
        generic_arg(p);
        if !p.at(T![>]) && !p.expect(T![,]) {
            break;
        }
    }
    p.expect(T![>]);
    m.complete(p, GENERIC_ARG_LIST);
}

pub(super) fn const_arg(p: &mut Parser) {
    let m = p.start();
    // FIXME: duplicates the code below
    match p.current() {
        T!['{'] => {
            expressions::block_expr(p);
            m.complete(p, CONST_ARG);
        }
        k if k.is_literal() => {
            expressions::literal(p);
            m.complete(p, CONST_ARG);
        }
        T![true] | T![false] => {
            expressions::literal(p);
            m.complete(p, CONST_ARG);
        }
        T![-] => {
            let lm = p.start();
            p.bump(T![-]);
            expressions::literal(p);
            lm.complete(p, PREFIX_EXPR);
            m.complete(p, CONST_ARG);
        }
        _ => {
            let lm = p.start();
            paths::use_path(p);
            lm.complete(p, PATH_EXPR);
            m.complete(p, CONST_ARG);
        }
    }
}

// test type_arg
// type A = B<'static, i32, 1, { 2 }, Item=u64, true, false>;
fn generic_arg(p: &mut Parser) {
    let m = p.start();
    match p.current() {
        LIFETIME_IDENT => {
            lifetime(p);
            m.complete(p, LIFETIME_ARG);
        }
        // test associated_type_bounds
        // fn print_all<T: Iterator<Item: Display>>(printables: T) {}
        IDENT if p.nth(1) == T![:] && p.nth(2) != T![:] => {
            name_ref(p);
            type_params::bounds(p);
            m.complete(p, ASSOC_TYPE_ARG);
        }
        IDENT if p.nth(1) == T![=] => {
            name_ref(p);
            p.bump_any();
            types::type_(p);
            m.complete(p, ASSOC_TYPE_ARG);
        }
        T!['{'] => {
            expressions::block_expr(p);
            m.complete(p, CONST_ARG);
        }
        k if k.is_literal() => {
            expressions::literal(p);
            m.complete(p, CONST_ARG);
        }
        T![true] | T![false] => {
            expressions::literal(p);
            m.complete(p, CONST_ARG);
        }
        // test const_generic_negated_literal
        // fn f() { S::<-1> }
        T![-] => {
            let lm = p.start();
            p.bump(T![-]);
            expressions::literal(p);
            lm.complete(p, PREFIX_EXPR);
            m.complete(p, CONST_ARG);
        }
        _ => {
            types::type_(p);
            m.complete(p, TYPE_ARG);
        }
    }
}
