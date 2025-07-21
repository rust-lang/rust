mod adt;
mod consts;
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
pub(super) fn mod_contents(p: &mut Parser<'_>, stop_on_r_curly: bool) {
    attributes::inner_attrs(p);
    while !(p.at(EOF) || (p.at(T!['}']) && stop_on_r_curly)) {
        // We can set `is_in_extern=true`, because it only allows `safe fn`, and there is no ambiguity here.
        item_or_macro(p, stop_on_r_curly, true);
    }
}

pub(super) const ITEM_RECOVERY_SET: TokenSet = TokenSet::new(&[
    T![fn],
    T![struct],
    T![enum],
    T![impl],
    T![trait],
    T![const],
    T![async],
    T![unsafe],
    T![extern],
    T![static],
    T![let],
    T![mod],
    T![pub],
    T![crate],
    T![use],
    T![macro],
    T![;],
]);

pub(super) fn item_or_macro(p: &mut Parser<'_>, stop_on_r_curly: bool, is_in_extern: bool) {
    let m = p.start();
    attributes::outer_attrs(p);

    let m = match opt_item(p, m, is_in_extern) {
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

    // test macro_rules_as_macro_name
    // macro_rules! {}
    // macro_rules! ();
    // macro_rules! [];
    // fn main() {
    //     let foo = macro_rules!();
    // }

    // test_err macro_rules_as_macro_name
    // macro_rules! {};
    // macro_rules! ()
    // macro_rules! []
    if paths::is_use_path_start(p) {
        paths::use_path(p);
        // Do not create a MACRO_CALL node here if this isn't a macro call, this causes problems with completion.

        // test_err path_item_without_excl
        // foo
        if p.at(T![!]) {
            macro_call(p, m);
            return;
        } else {
            m.complete(p, ERROR);
            p.error("expected an item");
            return;
        }
    }

    m.abandon(p);
    match p.current() {
        T!['{'] => error_block(p, "expected an item"),
        T!['}'] if !stop_on_r_curly => {
            let e = p.start();
            p.error("unmatched `}`");
            p.bump(T!['}']);
            e.complete(p, ERROR);
        }
        EOF | T!['}'] => p.error("expected an item"),
        T![let] => error_let_stmt(p, "expected an item"),
        _ => p.err_and_bump("expected an item"),
    }
}

/// Try to parse an item, completing `m` in case of success.
pub(super) fn opt_item(p: &mut Parser<'_>, m: Marker, is_in_extern: bool) -> Result<(), Marker> {
    // test_err pub_expr
    // fn foo() { pub 92; }
    let has_visibility = opt_visibility(p, false);

    let m = match opt_item_without_modifiers(p, m) {
        Ok(()) => return Ok(()),
        Err(m) => m,
    };

    let mut has_mods = false;
    let mut has_extern = false;

    // modifiers
    if p.at(T![const]) && p.nth(1) != T!['{'] {
        p.eat(T![const]);
        has_mods = true;
    }

    // test_err async_without_semicolon
    // fn foo() { let _ = async {} }
    if p.at(T![async])
        && (!matches!(p.nth(1), T!['{'] | T![gen] | T![move] | T![|])
            || matches!((p.nth(1), p.nth(2)), (T![gen], T![fn])))
    {
        p.eat(T![async]);
        has_mods = true;
    }

    // test_err gen_fn 2021
    // gen fn gen_fn() {}
    // async gen fn async_gen_fn() {}
    if p.at(T![gen]) && p.nth(1) == T![fn] {
        p.eat(T![gen]);
        has_mods = true;
    }

    // test_err unsafe_block_in_mod
    // fn foo(){} unsafe { } fn bar(){}
    if p.at(T![unsafe]) && p.nth(1) != T!['{'] {
        p.eat(T![unsafe]);
        has_mods = true;
    }

    // test safe_outside_of_extern
    // fn foo() { safe = true; }
    if is_in_extern && p.at_contextual_kw(T![safe]) {
        p.eat_contextual_kw(T![safe]);
        has_mods = true;
    }

    if p.at(T![extern]) {
        has_extern = true;
        has_mods = true;
        abi(p);
    }
    if p.at_contextual_kw(T![auto]) && p.nth(1) == T![trait] {
        p.bump_remap(T![auto]);
        has_mods = true;
    }

    // test default_item
    // default impl T for Foo {}
    if p.at_contextual_kw(T![default]) {
        match p.nth(1) {
            T![fn] | T![type] | T![const] | T![impl] => {
                p.bump_remap(T![default]);
                has_mods = true;
            }
            // test default_unsafe_item
            // default unsafe impl T for Foo {
            //     default unsafe fn foo() {}
            // }
            T![unsafe] if matches!(p.nth(2), T![impl] | T![fn]) => {
                p.bump_remap(T![default]);
                p.bump(T![unsafe]);
                has_mods = true;
            }
            // test default_async_fn
            // impl T for Foo {
            //     default async fn foo() {}
            // }
            T![async]
                if p.nth_at(2, T![fn]) || (p.nth_at(2, T![unsafe]) && p.nth_at(3, T![fn])) =>
            {
                p.bump_remap(T![default]);
                p.bump(T![async]);

                // test default_async_unsafe_fn
                // impl T for Foo {
                //     default async unsafe fn foo() {}
                // }
                p.eat(T![unsafe]);

                has_mods = true;
            }
            _ => (),
        }
    }

    // items
    match p.current() {
        T![fn] => fn_(p, m),

        T![const] if p.nth(1) != T!['{'] => consts::konst(p, m),
        T![static] if matches!(p.nth(1), IDENT | T![_] | T![mut]) => consts::static_(p, m),

        T![trait] => traits::trait_(p, m),
        T![impl] => traits::impl_(p, m),

        T![type] => type_alias(p, m),

        // test extern_block
        // unsafe extern "C" {}
        // extern {}
        T!['{'] if has_extern => {
            extern_item_list(p);
            m.complete(p, EXTERN_BLOCK);
        }

        _ if has_visibility || has_mods => {
            if has_mods {
                p.error("expected fn, trait or impl");
            } else {
                p.error("expected an item");
            }
            m.complete(p, ERROR);
        }

        _ => return Err(m),
    }
    Ok(())
}

fn opt_item_without_modifiers(p: &mut Parser<'_>, m: Marker) -> Result<(), Marker> {
    let la = p.nth(1);
    match p.current() {
        T![extern] if la == T![crate] => extern_crate(p, m),
        T![use] => use_item::use_(p, m),
        T![mod] => mod_item(p, m),

        T![type] => type_alias(p, m),
        T![struct] => adt::strukt(p, m),
        T![enum] => adt::enum_(p, m),
        IDENT if p.at_contextual_kw(T![union]) && p.nth(1) == IDENT => adt::union(p, m),

        T![macro] => macro_def(p, m),
        // check if current token is "macro_rules" followed by "!" followed by an identifier
        IDENT if p.at_contextual_kw(T![macro_rules]) && p.nth_at(1, BANG) && p.nth_at(2, IDENT) => {
            macro_rules(p, m)
        }

        T![const] if (la == IDENT || la == T![_] || la == T![mut]) => consts::konst(p, m),
        T![static] if (la == IDENT || la == T![_] || la == T![mut]) => consts::static_(p, m),

        IDENT
            if p.at_contextual_kw(T![builtin])
                && p.nth_at(1, T![#])
                && p.nth_at_contextual_kw(2, T![global_asm]) =>
        {
            p.bump_remap(T![builtin]);
            p.bump(T![#]);
            p.bump_remap(T![global_asm]);
            // test global_asm
            // builtin#global_asm("")
            expressions::parse_asm_expr(p, m);
        }

        _ => return Err(m),
    };
    Ok(())
}

// test extern_crate
// extern crate foo;
// extern crate self;
fn extern_crate(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![extern]);
    p.bump(T![crate]);

    name_ref_or_self(p);

    // test extern_crate_rename
    // extern crate foo as bar;
    // extern crate self as bar;
    opt_rename(p);
    p.expect(T![;]);
    m.complete(p, EXTERN_CRATE);
}

// test mod_item
// mod a;
pub(crate) fn mod_item(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![mod]);
    name(p);
    if p.at(T!['{']) {
        // test mod_item_curly
        // mod b { }
        item_list(p);
    } else if !p.eat(T![;]) {
        p.error("expected `;` or `{`");
    }
    m.complete(p, MODULE);
}

// test type_alias
// type Foo = Bar;
fn type_alias(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![type]);

    name(p);

    // test type_item_type_params
    // type Result<T> = ();
    generic_params::opt_generic_param_list(p);

    if p.at(T![:]) {
        generic_params::bounds(p);
    }

    // test type_item_where_clause_deprecated
    // type Foo where Foo: Copy = ();
    generic_params::opt_where_clause(p);
    if p.eat(T![=]) {
        types::type_(p);
    }

    // test type_item_where_clause
    // type Foo = () where Foo: Copy;
    generic_params::opt_where_clause(p);

    p.expect(T![;]);
    m.complete(p, TYPE_ALIAS);
}

pub(crate) fn item_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, ITEM_LIST);
}

pub(crate) fn extern_item_list(p: &mut Parser<'_>) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump(T!['{']);
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, EXTERN_ITEM_LIST);
}

// test try_macro_rules 2015
// macro_rules! try { () => {} }
fn macro_rules(p: &mut Parser<'_>, m: Marker) {
    assert!(p.at_contextual_kw(T![macro_rules]));
    p.bump_remap(T![macro_rules]);
    p.expect(T![!]);

    name(p);

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
// macro m($i:ident) {}
fn macro_def(p: &mut Parser<'_>, m: Marker) {
    p.expect(T![macro]);
    name_r(p, ITEM_RECOVERY_SET);
    if p.at(T!['{']) {
        // test macro_def_curly
        // macro m { ($i:ident) => {} }
        token_tree(p);
    } else if p.at(T!['(']) {
        token_tree(p);
        match p.current() {
            T!['{'] | T!['['] | T!['('] => token_tree(p),
            _ => p.error("expected `{`, `[`, `(`"),
        }
    } else {
        p.error("unmatched `(`");
    }

    m.complete(p, MACRO_DEF);
}

// test fn_
// fn foo() {}
fn fn_(p: &mut Parser<'_>, m: Marker) {
    p.bump(T![fn]);

    name_r(p, ITEM_RECOVERY_SET);
    // test function_type_params
    // fn foo<T: Clone + Copy>(){}
    generic_params::opt_generic_param_list(p);

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
    generic_params::opt_where_clause(p);

    // test fn_decl
    // trait T { fn foo(); }
    if !p.eat(T![;]) {
        expressions::block_expr(p);
    }
    m.complete(p, FN);
}

fn macro_call(p: &mut Parser<'_>, m: Marker) {
    assert!(p.at(T![!]));
    match macro_call_after_excl(p) {
        BlockLike::Block => (),
        BlockLike::NotBlock => {
            p.expect(T![;]);
        }
    }
    m.complete(p, MACRO_CALL);
}

pub(super) fn macro_call_after_excl(p: &mut Parser<'_>) -> BlockLike {
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

pub(crate) fn token_tree(p: &mut Parser<'_>) {
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
