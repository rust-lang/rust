use super::*;

// test_err generic_arg_list_recover
// type T = T<0, ,T>;
pub(super) fn opt_generic_arg_list(p: &mut Parser<'_>, colon_colon_required: bool) {
    let m;
    if p.at(T![::]) && p.nth(2) == T![<] {
        m = p.start();
        p.bump(T![::]);
    } else if !colon_colon_required && p.at(T![<]) && p.nth(1) != T![=] {
        m = p.start();
    } else {
        return;
    }

    delimited(
        p,
        T![<],
        T![>],
        T![,],
        || "expected generic argument".into(),
        GENERIC_ARG_FIRST,
        generic_arg,
    );
    m.complete(p, GENERIC_ARG_LIST);
}

const GENERIC_ARG_FIRST: TokenSet = TokenSet::new(&[
    LIFETIME_IDENT,
    IDENT,
    T!['{'],
    T![true],
    T![false],
    T![-],
    INT_NUMBER,
    FLOAT_NUMBER,
    CHAR,
    BYTE,
    STRING,
    BYTE_STRING,
    C_STRING,
])
.union(types::TYPE_FIRST);

// Despite its name, it can also be used for generic param list.
const GENERIC_ARG_RECOVERY_SET: TokenSet = TokenSet::new(&[T![>], T![,]]);

// test generic_arg
// type T = S<i32>;
fn generic_arg(p: &mut Parser<'_>) -> bool {
    match p.current() {
        LIFETIME_IDENT if !p.nth_at(1, T![+]) => lifetime_arg(p),
        T!['{'] | T![true] | T![false] | T![-] => const_arg(p),
        k if k.is_literal() => const_arg(p),
        // test associated_type_bounds
        // fn print_all<T: Iterator<Item, Item::Item, Item::<true>, Item: Display, Item<'a> = Item>>(printables: T) {}

        // test macro_inside_generic_arg
        // type A = Foo<syn::Token![_]>;
        IDENT if [T![<], T![=], T![:]].contains(&p.nth(1)) && !p.nth_at(1, T![::]) => {
            let m = p.start();
            name_ref(p);
            opt_generic_arg_list(p, false);
            match p.current() {
                T![=] => {
                    p.bump_any();
                    if types::TYPE_FIRST.contains(p.current()) {
                        // test assoc_type_eq
                        // type T = StreamingIterator<Item<'a> = &'a T>;
                        types::type_(p);
                    } else if p.at_ts(GENERIC_ARG_RECOVERY_SET) {
                        // Although `const_arg()` recovers as expected, we want to
                        // handle those here to give the following message because
                        // we don't know whether this associated item is a type or
                        // const at this point.

                        // test_err recover_from_missing_assoc_item_binding
                        // fn f() -> impl Iterator<Item = , Item = > {}
                        p.error("missing associated item binding");
                    } else {
                        // test assoc_const_eq
                        // fn foo<F: Foo<N=3>>() {}
                        // const TEST: usize = 3;
                        // fn bar<F: Foo<N={TEST}>>() {}
                        const_arg(p);
                    }
                    m.complete(p, ASSOC_TYPE_ARG);
                }
                // test assoc_type_bound
                // type T = StreamingIterator<Item<'a>: Clone>;
                T![:] if !p.at(T![::]) => {
                    generic_params::bounds(p);
                    m.complete(p, ASSOC_TYPE_ARG);
                }
                _ => {
                    let m = m.complete(p, PATH_SEGMENT).precede(p).complete(p, PATH);
                    let m = paths::type_path_for_qualifier(p, m);
                    m.precede(p).complete(p, PATH_TYPE).precede(p).complete(p, TYPE_ARG);
                }
            }
        }
        IDENT if p.nth_at(1, T!['(']) => {
            let m = p.start();
            name_ref(p);
            if p.nth_at(1, T![..]) {
                let rtn = p.start();
                p.bump(T!['(']);
                p.bump(T![..]);
                p.expect(T![')']);
                rtn.complete(p, RETURN_TYPE_SYNTAX);
                // test return_type_syntax_assoc_type_bound
                // fn foo<T: Trait<method(..): Send>>() {}
                generic_params::bounds(p);
                m.complete(p, ASSOC_TYPE_ARG);
            } else {
                params::param_list_fn_trait(p);
                // test bare_dyn_types_with_paren_as_generic_args
                // type A = S<Fn(i32)>;
                // type A = S<Fn(i32) + Send>;
                // type B = S<Fn(i32) -> i32>;
                // type C = S<Fn(i32) -> i32 + Send>;
                opt_ret_type(p);
                let m = m.complete(p, PATH_SEGMENT).precede(p).complete(p, PATH);
                let m = paths::type_path_for_qualifier(p, m);
                let m = m.precede(p).complete(p, PATH_TYPE);
                let m = types::opt_type_bounds_as_dyn_trait_type(p, m);
                m.precede(p).complete(p, TYPE_ARG);
            }
        }
        _ if p.at_ts(types::TYPE_FIRST) => type_arg(p),
        _ => return false,
    }
    true
}

// test lifetime_arg
// type T = S<'static>;
fn lifetime_arg(p: &mut Parser<'_>) {
    let m = p.start();
    lifetime(p);
    m.complete(p, LIFETIME_ARG);
}

pub(super) fn const_arg_expr(p: &mut Parser<'_>) {
    // The tests in here are really for `const_arg`, which wraps the content
    // CONST_ARG.
    match p.current() {
        // test const_arg_block
        // type T = S<{90 + 2}>;
        T!['{'] => {
            expressions::block_expr(p);
        }
        // test const_arg_literal
        // type T = S<"hello", 0xdeadbeef>;
        k if k.is_literal() => {
            expressions::literal(p);
        }
        // test const_arg_bool_literal
        // type T = S<true>;
        T![true] | T![false] => {
            expressions::literal(p);
        }
        // test const_arg_negative_number
        // type T = S<-92>;
        T![-] => {
            let lm = p.start();
            p.bump(T![-]);
            expressions::literal(p);
            lm.complete(p, PREFIX_EXPR);
        }
        _ if paths::is_use_path_start(p) => {
            // This shouldn't be hit by `const_arg`
            let lm = p.start();
            paths::use_path(p);
            lm.complete(p, PATH_EXPR);
        }
        _ => {
            // test_err recover_from_missing_const_default
            // struct A<const N: i32 = , const M: i32 =>;
            p.err_recover("expected a generic const argument", GENERIC_ARG_RECOVERY_SET);
        }
    }
}

// test const_arg
// type T = S<92>;
pub(super) fn const_arg(p: &mut Parser<'_>) {
    let m = p.start();
    const_arg_expr(p);
    m.complete(p, CONST_ARG);
}

fn type_arg(p: &mut Parser<'_>) {
    let m = p.start();
    types::type_(p);
    m.complete(p, TYPE_ARG);
}
