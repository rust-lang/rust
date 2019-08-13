mod consts;
mod nominal;
mod traits;
mod use_item;

pub(crate) use self::{
    expressions::{match_arm_list, named_field_list},
    nominal::{enum_variant_list, named_field_def_list},
    traits::{impl_item_list, trait_item_list},
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
    attributes::inner_attributes(p);
    while !p.at(EOF) && !(stop_on_r_curly && p.at(T!['}'])) {
        item_or_macro(p, stop_on_r_curly, ItemFlavor::Mod)
    }
}

pub(super) enum ItemFlavor {
    Mod,
    Trait,
}

pub(super) const ITEM_RECOVERY_SET: TokenSet = token_set![
    FN_KW, STRUCT_KW, ENUM_KW, IMPL_KW, TRAIT_KW, CONST_KW, STATIC_KW, LET_KW, MOD_KW, PUB_KW,
    CRATE_KW
];

pub(super) fn item_or_macro(p: &mut Parser, stop_on_r_curly: bool, flavor: ItemFlavor) {
    let m = p.start();
    attributes::outer_attributes(p);
    let m = match maybe_item(p, m, flavor) {
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
            p.bump();
            e.complete(p, ERROR);
        } else if !p.at(EOF) && !p.at(T!['}']) {
            p.err_and_bump("expected an item");
        } else {
            p.error("expected an item");
        }
    }
}

pub(super) fn maybe_item(p: &mut Parser, m: Marker, flavor: ItemFlavor) -> Result<(), Marker> {
    // test_err pub_expr
    // fn foo() { pub 92; }
    let has_visibility = opt_visibility(p);

    let m = match items_without_modifiers(p, m) {
        Ok(()) => return Ok(()),
        Err(m) => m,
    };

    let mut has_mods = false;

    // modifiers
    has_mods |= p.eat(T![const]);

    // test_err unsafe_block_in_mod
    // fn foo(){} unsafe { } fn bar(){}
    if p.at(T![unsafe]) && p.nth(1) != T!['{'] {
        p.eat(T![unsafe]);
        has_mods = true;
    }

    // test_err async_without_semicolon
    // fn foo() { let _ = async {} }
    if p.at(T![async]) && p.nth(1) != T!['{'] && p.nth(1) != T![move] && p.nth(1) != T![|] {
        p.eat(T![async]);
        has_mods = true;
    }

    if p.at(T![extern]) {
        has_mods = true;
        abi(p);
    }
    if p.at(IDENT) && p.at_contextual_kw("auto") && p.nth(1) == T![trait] {
        p.bump_remap(T![auto]);
        has_mods = true;
    }

    if p.at(IDENT)
        && p.at_contextual_kw("default")
        && (match p.nth(1) {
            T![impl] => true,
            T![fn] | T![type] => {
                if let ItemFlavor::Mod = flavor {
                    true
                } else {
                    false
                }
            }
            _ => false,
        })
    {
        p.bump_remap(T![default]);
        has_mods = true;
    }
    if p.at(IDENT) && p.at_contextual_kw("existential") && p.nth(1) == T![type] {
        p.bump_remap(T![existential]);
        has_mods = true;
    }

    // items
    match p.current() {
        // test async_fn
        // async fn foo() {}

        // test extern_fn
        // extern fn foo() {}

        // test const_fn
        // const fn foo() {}

        // test const_unsafe_fn
        // const unsafe fn foo() {}

        // test unsafe_extern_fn
        // unsafe extern "C" fn foo() {}

        // test unsafe_fn
        // unsafe fn foo() {}

        // test combined_fns
        // unsafe async fn foo() {}
        // const unsafe fn bar() {}

        // test_err wrong_order_fns
        // async unsafe fn foo() {}
        // unsafe const fn bar() {}
        T![fn] => {
            fn_def(p, flavor);
            m.complete(p, FN_DEF);
        }

        // test unsafe_trait
        // unsafe trait T {}

        // test auto_trait
        // auto trait T {}

        // test unsafe_auto_trait
        // unsafe auto trait T {}
        T![trait] => {
            traits::trait_def(p);
            m.complete(p, TRAIT_DEF);
        }

        // test unsafe_impl
        // unsafe impl Foo {}

        // test default_impl
        // default impl Foo {}

        // test_err default_fn_type
        // trait T {
        //     default type T = Bar;
        //     default fn foo() {}
        // }

        // test default_fn_type
        // impl T for Foo {
        //     default type T = Bar;
        //     default fn foo() {}
        // }

        // test unsafe_default_impl
        // unsafe default impl Foo {}
        T![impl] => {
            traits::impl_block(p);
            m.complete(p, IMPL_BLOCK);
        }

        // test existential_type
        // existential type Foo: Fn() -> usize;
        T![type] => {
            type_def(p, m);
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
        T![extern] if la == T![crate] => extern_crate_item(p, m),
        T![type] => {
            type_def(p, m);
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
            nominal::struct_def(p, m, T![struct]);
        }
        IDENT if p.at_contextual_kw("union") && p.nth(1) == IDENT => {
            // test union_items
            // union Foo {}
            // union Foo {
            //     a: i32,
            //     b: f32,
            // }
            nominal::struct_def(p, m, T![union]);
        }
        T![enum] => nominal::enum_def(p, m),
        T![use] => use_item::use_item(p, m),
        T![const] if (la == IDENT || la == T![mut]) => consts::const_def(p, m),
        T![static] => consts::static_def(p, m),
        // test extern_block
        // extern {}
        T![extern]
            if la == T!['{'] || ((la == STRING || la == RAW_STRING) && p.nth(2) == T!['{']) =>
        {
            abi(p);
            extern_item_list(p);
            m.complete(p, EXTERN_BLOCK);
        }
        _ => return Err(m),
    };
    Ok(())
}

fn extern_crate_item(p: &mut Parser, m: Marker) {
    assert!(p.at(T![extern]));
    p.bump();
    assert!(p.at(T![crate]));
    p.bump();
    name_ref(p);
    opt_alias(p);
    p.expect(T![;]);
    m.complete(p, EXTERN_CRATE_ITEM);
}

pub(crate) fn extern_item_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump();
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, EXTERN_ITEM_LIST);
}

fn fn_def(p: &mut Parser, flavor: ItemFlavor) {
    assert!(p.at(T![fn]));
    p.bump();

    name_r(p, ITEM_RECOVERY_SET);
    // test function_type_params
    // fn foo<T: Clone + Copy>(){}
    type_params::opt_type_param_list(p);

    if p.at(T!['(']) {
        match flavor {
            ItemFlavor::Mod => params::param_list(p),
            ItemFlavor::Trait => params::param_list_opt_patterns(p),
        }
    } else {
        p.error("expected function arguments");
    }
    // test function_ret_type
    // fn foo() {}
    // fn bar() -> () {}
    opt_fn_ret_type(p);

    // test function_where_clause
    // fn foo<T>() where T: Copy {}
    type_params::opt_where_clause(p);

    // test fn_decl
    // trait T { fn foo(); }
    if p.at(T![;]) {
        p.bump();
    } else {
        expressions::block(p)
    }
}

// test type_item
// type Foo = Bar;
fn type_def(p: &mut Parser, m: Marker) {
    assert!(p.at(T![type]));
    p.bump();

    name(p);

    // test type_item_type_params
    // type Result<T> = ();
    type_params::opt_type_param_list(p);

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
    m.complete(p, TYPE_ALIAS_DEF);
}

pub(crate) fn mod_item(p: &mut Parser, m: Marker) {
    assert!(p.at(T![mod]));
    p.bump();

    name(p);
    if p.at(T!['{']) {
        mod_item_list(p);
    } else if !p.eat(T![;]) {
        p.error("expected `;` or `{`");
    }
    m.complete(p, MODULE);
}

pub(crate) fn mod_item_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump();
    mod_contents(p, true);
    p.expect(T!['}']);
    m.complete(p, ITEM_LIST);
}

fn macro_call(p: &mut Parser) -> BlockLike {
    assert!(paths::is_use_path_start(p));
    paths::use_path(p);
    macro_call_after_excl(p)
}

pub(super) fn macro_call_after_excl(p: &mut Parser) -> BlockLike {
    p.expect(T![!]);
    if p.at(IDENT) {
        name(p);
    }
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
    p.bump();
    while !p.at(EOF) && !p.at(closing_paren_kind) {
        match p.current() {
            T!['{'] | T!['('] | T!['['] => token_tree(p),
            T!['}'] => {
                p.error("unmatched `}`");
                m.complete(p, TOKEN_TREE);
                return;
            }
            T![')'] | T![']'] => p.err_and_bump("unmatched brace"),
            _ => p.bump_raw(),
        }
    }
    p.expect(closing_paren_kind);
    m.complete(p, TOKEN_TREE);
}
