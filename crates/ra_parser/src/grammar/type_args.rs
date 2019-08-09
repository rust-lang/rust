use super::*;

pub(super) fn opt_type_arg_list(p: &mut Parser, colon_colon_required: bool) {
    let m;
    match (colon_colon_required, p.nth(0), p.nth(1)) {
        (_, T![::], T![<]) => {
            m = p.start();
            p.bump();
            p.bump();
        }
        (false, T![<], T![=]) => return,
        (false, T![<], _) => {
            m = p.start();
            p.bump();
        }
        _ => return,
    };

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
// type A = B<'static, i32, Item=u64>;
fn type_arg(p: &mut Parser) {
    let m = p.start();
    match p.current() {
        LIFETIME => {
            p.bump();
            m.complete(p, LIFETIME_ARG);
        }
        // test associated_type_bounds
        // fn print_all<T: Iterator<Item: Display>>(printables: T) {}
        IDENT if p.nth(1) == T![:] => {
            name_ref(p);
            type_params::bounds(p);
            m.complete(p, ASSOC_TYPE_ARG);
        }
        IDENT if p.nth(1) == T![=] => {
            name_ref(p);
            p.bump();
            types::type_(p);
            m.complete(p, ASSOC_TYPE_ARG);
        }
        _ => {
            types::type_(p);
            m.complete(p, TYPE_ARG);
        }
    }
}
