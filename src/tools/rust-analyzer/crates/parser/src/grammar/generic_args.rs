use super::*;

// test_err generic_arg_list_recover_expr
// const _: () = T::<0, ,T>;
// const _: () = T::<0, ,T>();
pub(super) fn opt_generic_arg_list_expr(p: &mut Parser<'_>) {
    let m;
    if p.at(T![::]) && p.nth(2) == T![<] {
        m = p.start();
        p.bump(T![::]);
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

pub(crate) const GENERIC_ARG_FIRST: TokenSet = TokenSet::new(&[
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
// type T = S<i32, dyn T, fn()>;
pub(crate) fn generic_arg(p: &mut Parser<'_>) -> bool {
    match p.current() {
        LIFETIME_IDENT if !p.nth_at(1, T![+]) => lifetime_arg(p),
        T!['{'] | T![true] | T![false] | T![-] => const_arg(p),
        k if k.is_literal() => const_arg(p),
        // test generic_arg_bounds
        // type Plain = Foo<Item, Item::Item, Item: Bound, Item = Item>;
        // type GenericArgs = Foo<Item<T>, Item::<T>, Item<T>: Bound, Item::<T>: Bound, Item<T> = Item, Item::<T> = Item>;
        // type ParenthesizedArgs = Foo<Item(T), Item::(T), Item(T): Bound, Item::(T): Bound, Item(T) = Item, Item::(T) = Item>;
        // type RTN = Foo<Item(..), Item(..), Item(..): Bound, Item(..): Bound, Item(..) = Item, Item(..) = Item>;

        // test edition_2015_dyn_prefix_inside_generic_arg 2015
        // type A = Foo<dyn T>;
        T![ident] if !p.edition().at_least_2018() && types::is_dyn_weak(p) => type_arg(p),
        // test macro_inside_generic_arg
        // type A = Foo<syn::Token![_]>;
        k if PATH_NAME_REF_KINDS.contains(k) => {
            let m = p.start();
            name_ref_mod_path(p);
            paths::opt_path_type_args(p);
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
                // type T = StreamingIterator<Item(T): Clone>;
                T![:] if !p.at(T![::]) => {
                    generic_params::bounds(p);
                    m.complete(p, ASSOC_TYPE_ARG);
                }
                // Turned out to be just a normal path type (mirror `path_or_macro_type`)
                _ => {
                    let m = m.complete(p, PATH_SEGMENT).precede(p).complete(p, PATH);
                    let m = paths::type_path_for_qualifier(p, m);
                    let m = if p.at(T![!]) && !p.at(T![!=]) {
                        let m = m.precede(p);
                        items::macro_call_after_excl(p);
                        m.complete(p, MACRO_CALL).precede(p).complete(p, MACRO_TYPE)
                    } else {
                        m.precede(p).complete(p, PATH_TYPE)
                    };
                    types::opt_type_bounds_as_dyn_trait_type(p, m).precede(p).complete(p, TYPE_ARG);
                }
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
        _ if paths::is_path_start(p) => {
            // This shouldn't be hit by `const_arg`
            let lm = p.start();
            paths::expr_path(p);
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

pub(crate) fn type_arg(p: &mut Parser<'_>) {
    let m = p.start();
    types::type_(p);
    m.complete(p, TYPE_ARG);
}
