mod consts;
mod adt;
mod traits;
mod use_item;

pub(crate) use self::{
    adt::{record_field_list, variant_list},
    expressions::{match_arm_list, record_expr_field_list},
    traits::assoc_item_list,
    use_item::use_tree_list,
};
use super::*;

// test mod_contents
// fn foo() {}
// macro_rules! foo {}
// foo::bar!();
// super::baz! {}
// struct S;
pub(super) fn mod_contents(p: &mut Parser, stop_on_r_curly: bool) {
    attributes::inner_attrs(p);
    while !(stop_on_r_curly && p.at(T!['}']) || p.at(EOF)) {
        item_or_macro(p, stop_on_r_curly)
    }
}

pub(super) const ITEM_RECOVERY_SET: TokenSet = TokenSet::new(&[
    T![fn],
    T![struct],
    T![enum],
    T![impl],
    T![trait],
    T![const],
    T![static],
    T![let],
    T![mod],
    T![pub],
    T![crate],
    T![use],
    T![macro],
    T![;],
]);

pub(super) fn item_or_macro(p: &mut Parser, stop_on_r_curly: bool) {
    let m = p.start();
    attributes::outer_attrs(p);
    let m = match maybe_item(p, m) {
        Ok(()) => {
            if p.at(T![;]) {
                p.err_and_bump(
                    "expected item, found `;`\n\
                     consider removing this semicolon",
                );
            }
            return;
        }
        Err(m) => m,
    };
    if paths::is_use_path_start(p) {
        match macro_call(p) {
            BlockLike::Block => (),
            BlockLike::NotBlock => {
                p.expect(T![;]);
            }
        }
        m.complete(p, MACRO_CALL);
    } else {
        m.abandon(p);
        if p.at(T!['{']) {
            error_block(p, "expected an item");
        } else if p.at(T!['}']) && !stop_on_r_curly {
            let e = p.start();
            p.error("unmatched `}`");
            p.bump(T!['}']);
            e.complete(p, ERROR);
        } else if !p.at(EOF) && !p.at(T!['}']) {
            p.err_and_bump("expected an item");
        } else {
            p.error("expected an item");
        }
    }
}

/// Try to parse an item, completing `m` in case of success.
pub(super) fn maybe_item(p: &mut Parser, m: Marker) -> Result<(), Marker> {
    // test_err pub_expr
    // fn foo() { pub 92; }
    let has_visibility = opt_visibility(p);

    let m = match items_without_modifiers(p, m) {
        Ok(()) => return Ok(()),
        Err(m) => m,
    };

    let mut has_mods = false;

    // modifiers
    if p.at(T![const]) && p.nth(1) != T!['{'] {
        p.eat(T![const]);
        has_mods = true;
    }

    // test_err async_without_semicolon
    // fn foo() { let _ = async {} }
    if p.at(T![async]) && p.nth(1) != T!['{'] && p.nth(1) != T![move] && p.nth(1) != T![|] {
        p.eat(T![async]);
        has_mods = true;
    }

    // test_err unsafe_block_in_mod
    // fn foo(){} unsafe { } fn bar(){}
    if p.at(T![unsafe]) && p.nth(1) != T!['{'] {
        p.eat(T![unsafe]);
        has_mods = true;
    }

    if p.at(T![extern]) && p.nth(1) != T!['{'] && (p.nth(1) != STRING || p.nth(2) != T!['{']) {
        has_mods = true;
        abi(p);
    }
    if p.at(IDENT) && p.at_contextual_kw("auto") && p.nth(1) == T![trait] {
        p.bump_remap(T![auto]);
        has_mods = true;
    }

    // test default_item
    // default impl T for Foo {}
    if p.at(IDENT) && p.at_contextual_kw("default") {
        match p.nth(1) {
            T![fn] | T![type] | T![const] | T![impl] => {
                p.bump_remap(T![default]);
                has_mods = true;
            }
            T![unsafe] => {
                // test default_unsafe_item
                // default unsafe impl T for Foo {
                //     default unsafe fn foo() {}
                // }
                if matches!(p.nth(2), T![impl] | T![fn]) {
                    p.bump_remap(T![default]);
                    p.bump(T![unsafe]);
                    has_mods = true;
                }
            }
            T![async] => {
                // test default_async_fn
                // impl T for Foo {
                //     default async fn foo() {}
                // }

                // test default_async_unsafe_fn
                // impl T for Foo {
                //     default async unsafe fn foo() {}
                // }
                let mut maybe_fn = p.nth(2);
                let is_unsafe = if matches!(maybe_fn, T![unsafe]) {
                    maybe_fn = p.nth(3);
                    true
                } else {
                    false
                };

                if matches!(maybe_fn, T![fn]) {
                    p.bump_remap(T![default]);
                    p.bump(T![async]);
                    if is_unsafe {
                        p.bump(T![unsafe])
                    }
                    has_mods = true;
                }
            }
            _ => (),
        }
    }

    // test existential_type
    // existential type Foo: Fn() -> usize;
    if p.at(IDENT) && p.at_contextual_kw("existential") && p.nth(1) == T![type] {
        p.bump_remap(T![existential]);
        has_mods = true;
    }

    // items
    match p.current() {
        // test fn
        // fn foo() {}
        T![fn] => {
            fn_(p);
            m.complete(p, FN);
        }

        // test trait
        // trait T {}
        T![trait] => {
            traits::trait_(p);
            m.complete(p, TRAIT);
        }

        T![const] if p.nth(1) != T!['{'] => {
            consts::konst(p, m);
        }

        // test impl
        // impl T for S {}
        T![impl] => {
            traits::impl_(p);
            m.complete(p, IMPL);
        }

        T![type] => {
            type_alias(p, m);
        }

        // unsafe extern "C" {}
        T![extern] => {
            abi(p);
            extern_item_list(p);
            m.complete(p, EXTERN_BLOCK);
        }

        _ => {
            if !has_visibility && !has_mods {
                return Err(m);
            } else {
                if has_mods {
                    p.error("expected existential, fn, trait or impl");
                } else {
                    p.error("expected an item");
                }
                m.complete(p, ERROR);
            }
        }
    }
    Ok(())
}

fn items_without_modifiers(p: &mut Parser, m: Marker) -> Result<(), Marker> {
    let la = p.nth(1);
    match p.current() {
        // test extern_crate
        // extern crate foo;
        T![extern] if la == T![crate] => extern_crate(p, m),
        T![type] => {
            type_alias(p, m);
        }
        T![mod] => mod_item(p, m),
        T![struct] => {
            // test struct_items
            // struct Foo;
            // struct Foo {}
            // struct Foo();
            // struct Foo(String, usize);
            // struct Foo {
            //     a: i32,
            //     b: f32,
            // }
            adt::strukt(p, m);
        }
        // test pub_macro_def
        // pub macro m($:ident) {}
        T![macro] => {
            macro_def(p, m);
        }
        IDENT if p.at_contextual_kw("macro_rules") && p.nth(1) == BANG => {
            macro_rules(p, m);
        }
        IDENT if p.at_contextual_kw("union") && p.nth(1) == IDENT => {
            // test union_items
            // union Foo {}
            // union Foo {
            //     a: i32,
            //     b: f32,
            // }
            adt::union(p, m);
        }
        T![enum] => adt::enum_(p, m),
        T![use] => use_item::use_(p, m),
        T![const] if (la == IDENT || la == T![_] || la == T![mut]) => consts::konst(p, m),
        T![static] => consts::static_(p, m),
        // test extern_block
        // extern {}
        T![extern] if la == T!['{'] || (la == STRING && p.nth(2) == T!['{']) => {
            abi(p);
            extern_item_list(p);
            m.complete(p, EXTERN_BLOCK);
        }
        _ => return Err(m),
    };
    Ok(())
}

fn extern_crate(p: &mut Parser, m: Marker) {
    assert!(p.at(T![extern]));
    p.bump(T![extern]);
    assert!(p.at(T![crate]));
    p.bump(T![crate]);

    if p.at(T![self]) {
        let m = p.start();
        p.bump(T![self]);
        m.complete(p, NAME_REF);
    } else {
        name_ref(p);
    }

    opt_rename(p);
    p.expect(T![;]);
    m.complete(p, EXTERN_CRATE);
}

pub(crate) fn extern_item_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, EXTERN_ITEM_LIST);
}

fn fn_(p: &mut Parser) {
    assert!(p.at(T![fn]));
    p.bump(T![fn]);

    name_r(p, ITEM_RECOVERY_SET);
    // test function_type_params
    // fn foo<T: Clone + Copy>(){}
    type_params::opt_generic_param_list(p);

    if p.at(T!['(']) {
        params::param_list_fn_def(p);
    } else {
        p.error("expected function arguments");
    }
    // test function_ret_type
    // fn foo() {}
    // fn bar() -> () {}
    opt_ret_type(p);

    // test function_where_clause
    // fn foo<T>() where T: Copy {}
    type_params::opt_where_clause(p);

    // test fn_decl
    // trait T { fn foo(); }
    if p.at(T![;]) {
        p.bump(T![;]);
    } else {
        expressions::block_expr(p)
    }
}

// test type_item
// type Foo = Bar;
fn type_alias(p: &mut Parser, m: Marker) {
    assert!(p.at(T![type]));
    p.bump(T![type]);

    name(p);

    // test type_item_type_params
    // type Result<T> = ();
    type_params::opt_generic_param_list(p);

    if p.at(T![:]) {
        type_params::bounds(p);
    }

    // test type_item_where_clause
    // type Foo where Foo: Copy = ();
    type_params::opt_where_clause(p);
    if p.eat(T![=]) {
        types::type_(p);
    }
    p.expect(T![;]);
    m.complete(p, TYPE_ALIAS);
}

pub(crate) fn mod_item(p: &mut Parser, m: Marker) {
    assert!(p.at(T![mod]));
    p.bump(T![mod]);

    name(p);
    if p.at(T!['{']) {
        item_list(p);
    } else if !p.eat(T![;]) {
        p.error("expected `;` or `{`");
    }
    m.complete(p, MODULE);
}

pub(crate) fn item_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, ITEM_LIST);
}

fn macro_rules(p: &mut Parser, m: Marker) {
    assert!(p.at_contextual_kw("macro_rules"));
    p.bump_remap(T![macro_rules]);
    p.expect(T![!]);

    if p.at(IDENT) {
        name(p);
    }
    // Special-case `macro_rules! try`.
    // This is a hack until we do proper edition support

    // test try_macro_rules
    // macro_rules! try { () => {} }
    if p.at(T![try]) {
        let m = p.start();
        p.bump_remap(IDENT);
        m.complete(p, NAME);
    }

    match p.current() {
        // test macro_rules_non_brace
        // macro_rules! m ( ($i:ident) => {} );
        // macro_rules! m [ ($i:ident) => {} ];
        T!['['] | T!['('] => {
            token_tree(p);
            p.expect(T![;]);
        }
        T!['{'] => token_tree(p),
        _ => p.error("expected `{`, `[`, `(`"),
    }
    m.complete(p, MACRO_RULES);
}

// test macro_def
// macro m { ($i:ident) => {} }
// macro m($i:ident) {}
fn macro_def(p: &mut Parser, m: Marker) {
    p.expect(T![macro]);
    name_r(p, ITEM_RECOVERY_SET);
    if p.at(T!['{']) {
        token_tree(p);
    } else if !p.at(T!['(']) {
        p.error("unmatched `(`");
    } else {
        let m = p.start();
        token_tree(p);
        match p.current() {
            T!['{'] | T!['['] | T!['('] => token_tree(p),
            _ => p.error("expected `{`, `[`, `(`"),
        }
        m.complete(p, TOKEN_TREE);
    }

    m.complete(p, MACRO_DEF);
}

fn macro_call(p: &mut Parser) -> BlockLike {
    assert!(paths::is_use_path_start(p));
    paths::use_path(p);
    macro_call_after_excl(p)
}

pub(super) fn macro_call_after_excl(p: &mut Parser) -> BlockLike {
    p.expect(T![!]);

    match p.current() {
        T!['{'] => {
            token_tree(p);
            BlockLike::Block
        }
        T!['('] | T!['['] => {
            token_tree(p);
            BlockLike::NotBlock
        }
        _ => {
            p.error("expected `{`, `[`, `(`");
            BlockLike::NotBlock
        }
    }
}

pub(crate) fn token_tree(p: &mut Parser) {
    let closing_paren_kind = match p.current() {
        T!['{'] => T!['}'],
        T!['('] => T![')'],
        T!['['] => T![']'],
        _ => unreachable!(),
    };
    let m = p.start();
    p.bump_any();
    while !p.at(EOF) && !p.at(closing_paren_kind) {
        match p.current() {
            T!['{'] | T!['('] | T!['['] => token_tree(p),
            T!['}'] => {
                p.error("unmatched `}`");
                m.complete(p, TOKEN_TREE);
                return;
            }
            T![')'] | T![']'] => p.err_and_bump("unmatched brace"),
            _ => p.bump_any(),
        }
    }
    p.expect(closing_paren_kind);
    m.complete(p, TOKEN_TREE);
}
