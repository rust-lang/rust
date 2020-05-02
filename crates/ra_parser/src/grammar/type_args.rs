//! FIXME: write short doc here

use super::*;

pub(super) fn opt_type_arg_list(p: &mut Parser, colon_colon_required: bool) {
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
        type_arg(p);
        if !p.at(T![>]) && !p.expect(T![,]) {
            break;
        }
    }
    p.expect(T![>]);
    m.complete(p, TYPE_ARG_LIST);
}

// test type_arg
// type A = B<'static, i32, 1, { 2 }, Item=u64>;
fn type_arg(p: &mut Parser) {
    let m = p.start();
    match p.current() {
        LIFETIME => {
            p.bump(LIFETIME);
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
            p.bump(k);
            m.complete(p, CONST_ARG);
        }
        _ => {
            types::type_(p);
            m.complete(p, TYPE_ARG);
        }
    }
}
