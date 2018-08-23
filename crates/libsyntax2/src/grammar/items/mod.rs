use super::*;

mod consts;
mod structs;
mod traits;
mod use_item;

// test mod_contents
// fn foo() {}
// macro_rules! foo {}
// foo::bar!();
// super::baz! {}
// struct S;
pub(super) fn mod_contents(p: &mut Parser, stop_on_r_curly: bool) {
    attributes::inner_attributes(p);
    while !p.at(EOF) && !(stop_on_r_curly && p.at(R_CURLY)) {
        item_or_macro(p, stop_on_r_curly, ItemFlavor::Mod)
    }
}

pub(super) enum ItemFlavor {
    Mod, Trait
}

pub(super) fn item_or_macro(p: &mut Parser, stop_on_r_curly: bool, flavor: ItemFlavor) {
    let m = p.start();
    match maybe_item(p, flavor) {
        MaybeItem::Item(kind) => {
            m.complete(p, kind);
        }
        MaybeItem::None => {
            if paths::is_path_start(p) {
                match macro_call(p) {
                    BlockLike::Block => (),
                    BlockLike::NotBlock => {
                        p.expect(SEMI);
                    }
                }
                m.complete(p, MACRO_CALL);
            } else {
                m.abandon(p);
                if p.at(L_CURLY) {
                    error_block(p, "expected an item");
                } else if !p.at(EOF) && !(stop_on_r_curly && p.at(R_CURLY)) {
                    p.err_and_bump("expected an item");
                } else {
                    p.error("expected an item");
                }
            }
        }
        MaybeItem::Modifiers => {
            p.error("expected fn, trait or impl");
            m.complete(p, ERROR);
        }
    }
}

pub(super) enum MaybeItem {
    None,
    Item(SyntaxKind),
    Modifiers,
}

pub(super) fn maybe_item(p: &mut Parser, flavor: ItemFlavor) -> MaybeItem {
    attributes::outer_attributes(p);
    opt_visibility(p);
    if let Some(kind) = items_without_modifiers(p) {
        return MaybeItem::Item(kind);
    }

    let mut has_mods = false;
    // modifiers
    has_mods |= p.eat(CONST_KW);

    // test unsafe_block_in_mod
    // fn foo(){} unsafe { } fn bar(){}
    if p.at(UNSAFE_KW) && p.nth(1) != L_CURLY {
        p.eat(UNSAFE_KW);
        has_mods = true;
    }
    if p.at(EXTERN_KW) {
        has_mods = true;
        abi(p);
    }
    if p.at(IDENT) && p.at_contextual_kw("auto") && p.nth(1) == TRAIT_KW {
        p.bump_remap(AUTO_KW);
        has_mods = true;
    }
    if p.at(IDENT) && p.at_contextual_kw("default") && p.nth(1) == IMPL_KW {
        p.bump_remap(DEFAULT_KW);
        has_mods = true;
    }

    // items
    let kind = match p.current() {
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
        FN_KW => {
            function(p, flavor);
            FN_DEF
        }

        // test unsafe_trait
        // unsafe trait T {}

        // test auto_trait
        // auto trait T {}

        // test unsafe_auto_trait
        // unsafe auto trait T {}
        TRAIT_KW => {
            traits::trait_def(p);
            TRAIT_DEF
        }

        // test unsafe_impl
        // unsafe impl Foo {}

        // test default_impl
        // default impl Foo {}

        // test unsafe_default_impl
        // unsafe default impl Foo {}
        IMPL_KW => {
            traits::impl_item(p);
            IMPL_ITEM
        }
        _ => return if has_mods {
            MaybeItem::Modifiers
        } else {
            MaybeItem::None
        }
    };

    MaybeItem::Item(kind)
}

fn items_without_modifiers(p: &mut Parser) -> Option<SyntaxKind> {
    let la = p.nth(1);
    let kind = match p.current() {
        // test extern_crate
        // extern crate foo;
        EXTERN_KW if la == CRATE_KW => {
            extern_crate_item(p);
            EXTERN_CRATE_ITEM
        }
        TYPE_KW => {
            type_def(p);
            TYPE_DEF
        }
        MOD_KW => {
            mod_item(p);
            MODULE
        }
        STRUCT_KW => {
            structs::struct_def(p);
            if p.at(SEMI) {
                p.err_and_bump(
                    "expected item, found `;`\n\
                     consider removing this semicolon"
                );
            }
            STRUCT_DEF
        }
        ENUM_KW => {
            structs::enum_def(p);
            ENUM_DEF
        }
        USE_KW => {
            use_item::use_item(p);
            USE_ITEM
        }
        CONST_KW if (la == IDENT || la == MUT_KW) => {
            consts::const_def(p);
            CONST_DEF
        }
        STATIC_KW => {
            consts::static_def(p);
            STATIC_DEF
        }
        // test extern_block
        // extern {}
        EXTERN_KW if la == L_CURLY || ((la == STRING || la == RAW_STRING) && p.nth(2) == L_CURLY) => {
            abi(p);
            extern_block(p);
            EXTERN_BLOCK_EXPR
        }
        _ => return None,
    };
    Some(kind)
}

fn extern_crate_item(p: &mut Parser) {
    assert!(p.at(EXTERN_KW));
    p.bump();
    assert!(p.at(CRATE_KW));
    p.bump();
    name(p);
    opt_alias(p);
    p.expect(SEMI);
}

fn extern_block(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    p.expect(R_CURLY);
}

fn function(p: &mut Parser, flavor: ItemFlavor) {
    assert!(p.at(FN_KW));
    p.bump();

    name(p);
    // test function_type_params
    // fn foo<T: Clone + Copy>(){}
    type_params::type_param_list(p);

    if p.at(L_PAREN) {
        match flavor {
            ItemFlavor::Mod =>
                params::param_list(p),
            ItemFlavor::Trait =>
                params::param_list_opt_patterns(p),
        }
    } else {
        p.error("expected function arguments");
    }
    // test function_ret_type
    // fn foo() {}
    // fn bar() -> () {}
    fn_ret_type(p);

    // test function_where_clause
    // fn foo<T>() where T: Copy {}
    type_params::where_clause(p);

    // test fn_decl
    // trait T { fn foo(); }
    if !p.eat(SEMI) {
        expressions::block(p);
    }
}

// test type_item
// type Foo = Bar;
fn type_def(p: &mut Parser) {
    assert!(p.at(TYPE_KW));
    p.bump();

    name(p);

    // test type_item_type_params
    // type Result<T> = ();
    type_params::type_param_list(p);

    if p.at(COLON) {
        type_params::bounds(p);
    }

    // test type_item_where_clause
    // type Foo where Foo: Copy = ();
    type_params::where_clause(p);

    if p.eat(EQ) {
        types::type_(p);
    }
    p.expect(SEMI);
}

fn mod_item(p: &mut Parser) {
    assert!(p.at(MOD_KW));
    p.bump();

    name(p);
    if !p.eat(SEMI) {
        if p.expect(L_CURLY) {
            mod_contents(p, true);
            p.expect(R_CURLY);
        }
    }
}

fn macro_call(p: &mut Parser) -> BlockLike {
    assert!(paths::is_path_start(p));
    paths::use_path(p);
    macro_call_after_excl(p)
}

pub(super) fn macro_call_after_excl(p: &mut Parser) -> BlockLike {
    p.expect(EXCL);
    p.eat(IDENT);
    let flavor = match p.current() {
        L_CURLY => {
            token_tree(p);
            BlockLike::Block
        }
        L_PAREN | L_BRACK => {
            token_tree(p);
            BlockLike::NotBlock
        }
        _ => {
            p.error("expected `{`, `[`, `(`");
            BlockLike::NotBlock
        },
    };

    flavor
}

pub(super) fn token_tree(p: &mut Parser) {
    let closing_paren_kind = match p.current() {
        L_CURLY => R_CURLY,
        L_PAREN => R_PAREN,
        L_BRACK => R_BRACK,
        _ => unreachable!(),
    };
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(closing_paren_kind) {
        match p.current() {
            L_CURLY | L_PAREN | L_BRACK => token_tree(p),
            R_CURLY | R_PAREN | R_BRACK => p.err_and_bump("unmatched brace"),
            _ => p.bump()
        }
    };
    p.expect(closing_paren_kind);
    m.complete(p, TOKEN_TREE);
}
