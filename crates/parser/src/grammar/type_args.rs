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
        // fn print_all<T: Iterator<Item, Item::Item, Item::<true>, Item: Display, Item<'a> = Item>>(printables: T) {}
        IDENT if [T![<], T![=], T![:]].contains(&p.nth(1)) => {
            let path_ty = p.start();
            let path = p.start();
            let path_seg = p.start();
            name_ref(p);
            opt_generic_arg_list(p, false);
            match p.current() {
                // NameRef<...> =
                T![=] => {
                    p.bump_any();
                    types::type_(p);

                    path_seg.abandon(p);
                    path.abandon(p);
                    path_ty.abandon(p);
                    m.complete(p, ASSOC_TYPE_ARG);
                }
                T![:] if p.nth(1) == T![:] => {
                    // NameRef::, this is a path type
                    path_seg.complete(p, PATH_SEGMENT);
                    let qual = path.complete(p, PATH);
                    opt_generic_arg_list(p, false);
                    paths::type_path_for_qualifier(p, qual);
                    path_ty.complete(p, PATH_TYPE);
                    m.complete(p, TYPE_ARG);
                }
                // NameRef<...>:
                T![:] => {
                    type_params::bounds(p);

                    path_seg.abandon(p);
                    path.abandon(p);
                    path_ty.abandon(p);
                    m.complete(p, ASSOC_TYPE_ARG);
                }
                // NameRef, this is a single segment path type
                _ => {
                    path_seg.complete(p, PATH_SEGMENT);
                    path.complete(p, PATH);
                    path_ty.complete(p, PATH_TYPE);
                    m.complete(p, TYPE_ARG);
                }
            }
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
