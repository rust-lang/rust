use super::*;

mod consts;
mod structs;
mod traits;
mod use_item;

pub(super) fn mod_contents(p: &mut Parser, stop_on_r_curly: bool) {
    attributes::inner_attributes(p);
    while !p.at(EOF) && !(stop_on_r_curly && p.at(R_CURLY)) {
        item(p);
    }
}

pub(super) const ITEM_FIRST: TokenSet =
    token_set![EXTERN_KW, MOD_KW, USE_KW, STRUCT_KW, ENUM_KW, FN_KW, PUB_KW, POUND];

pub(super) fn item(p: &mut Parser) {
    let item = p.start();
    attributes::outer_attributes(p);
    visibility(p);
    let la = p.nth(1);
    let item_kind = match p.current() {
        USE_KW => {
            use_item::use_item(p);
            USE_ITEM
        }
        // test extern_crate
        // extern crate foo;
        EXTERN_KW if la == CRATE_KW => {
            extern_crate_item(p);
            EXTERN_CRATE_ITEM
        }
        EXTERN_KW => {
            abi(p);
            match p.current() {
                // test extern_fn
                // extern fn foo() {}
                FN_KW => {
                    fn_item(p);
                    FN_ITEM
                }
                // test extern_block
                // extern {}
                L_CURLY => {
                    extern_block(p);
                    EXTERN_BLOCK
                }
                // test extern_struct
                // extern struct Foo;
                _ => {
                    item.abandon(p);
                    p.error("expected `fn` or `{`");
                    return;
                }
            }
        }
        STATIC_KW => {
            consts::static_item(p);
            STATIC_ITEM
        }
        CONST_KW => match p.nth(1) {
            // test const_fn
            // const fn foo() {}
            FN_KW => {
                p.bump();
                fn_item(p);
                FN_ITEM
            }
            // test const_unsafe_fn
            // const unsafe fn foo() {}
            UNSAFE_KW if p.nth(2) == FN_KW => {
                p.bump();
                p.bump();
                fn_item(p);
                FN_ITEM
            }
            _ => {
                consts::const_item(p);
                CONST_ITEM
            }
        },
        UNSAFE_KW => {
            p.bump();
            let la = p.nth(1);
            match p.current() {
                // test unsafe_trait
                // unsafe trait T {}
                TRAIT_KW => {
                    traits::trait_item(p);
                    TRAIT_ITEM
                }

                // test unsafe_auto_trait
                // unsafe auto trait T {}
                IDENT if p.at_contextual_kw("auto") && la == TRAIT_KW => {
                    p.bump_remap(AUTO_KW);
                    traits::trait_item(p);
                    TRAIT_ITEM
                }

                // test unsafe_impl
                // unsafe impl Foo {}
                IMPL_KW => {
                    traits::impl_item(p);
                    IMPL_ITEM
                }

                // test unsafe_default_impl
                // unsafe default impl Foo {}
                IDENT if p.at_contextual_kw("default") && la == IMPL_KW => {
                    p.bump_remap(DEFAULT_KW);
                    traits::impl_item(p);
                    IMPL_ITEM
                }

                // test unsafe_extern_fn
                // unsafe extern "C" fn foo() {}
                EXTERN_KW => {
                    abi(p);
                    if !p.at(FN_KW) {
                        item.abandon(p);
                        p.error("expected function");
                        return;
                    }
                    fn_item(p);
                    FN_ITEM
                }

                // test unsafe_fn
                // unsafe fn foo() {}
                FN_KW => {
                    fn_item(p);
                    FN_ITEM
                }

                t => {
                    item.abandon(p);
                    let message = "expected `trait`, `impl` or `fn`";

                    // test unsafe_block_in_mod
                    // fn foo(){} unsafe { } fn bar(){}
                    if t == L_CURLY {
                        error_block(p, message);
                    } else {
                        p.error(message);
                    }
                    return;
                }
            }
        }
        TRAIT_KW => {
            traits::trait_item(p);
            TRAIT_ITEM
        }
        // test auto_trait
        // auto trait T {}
        IDENT if p.at_contextual_kw("auto") && la == TRAIT_KW => {
            p.bump_remap(AUTO_KW);
            traits::trait_item(p);
            TRAIT_ITEM
        }
        IMPL_KW => {
            traits::impl_item(p);
            IMPL_ITEM
        }
        // test default_impl
        // default impl Foo {}
        IDENT if p.at_contextual_kw("default") && la == IMPL_KW => {
            p.bump_remap(DEFAULT_KW);
            traits::impl_item(p);
            IMPL_ITEM
        }

        FN_KW => {
            fn_item(p);
            FN_ITEM
        }
        TYPE_KW => {
            type_item(p);
            TYPE_ITEM
        }
        MOD_KW => {
            mod_item(p);
            MOD_ITEM
        }
        STRUCT_KW => {
            structs::struct_item(p);
            STRUCT_ITEM
        }
        ENUM_KW => {
            structs::enum_item(p);
            ENUM_ITEM
        }
        L_CURLY => {
            item.abandon(p);
            error_block(p, "expected item");
            return;
        }
        err_token => {
            item.abandon(p);
            let message = if err_token == SEMI {
                //TODO: if the item is incomplete, this message is misleading
                "expected item, found `;`\n\
                 consider removing this semicolon"
            } else {
                "expected item"
            };
            p.err_and_bump(message);
            return;
        }
    };
    item.complete(p, item_kind);
}

fn extern_crate_item(p: &mut Parser) {
    assert!(p.at(EXTERN_KW));
    p.bump();
    assert!(p.at(CRATE_KW));
    p.bump();
    name(p);
    alias(p);
    p.expect(SEMI);
}

fn extern_block(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    p.expect(R_CURLY);
}

fn fn_item(p: &mut Parser) {
    assert!(p.at(FN_KW));
    p.bump();

    name(p);
    // test fn_item_type_params
    // fn foo<T: Clone + Copy>(){}
    type_params::list(p);

    if p.at(L_PAREN) {
        params::list(p);
    } else {
        p.error("expected function arguments");
    }
    // test fn_item_ret_type
    // fn foo() {}
    // fn bar() -> () {}
    fn_ret_type(p);

    // test fn_item_where_clause
    // fn foo<T>() where T: Copy {}
    type_params::where_clause(p);

    expressions::block(p);
}

// test type_item
// type Foo = Bar;
fn type_item(p: &mut Parser) {
    assert!(p.at(TYPE_KW));
    p.bump();

    name(p);

    // test type_item_type_params
    // type Result<T> = ();
    type_params::list(p);

    // test type_item_where_clause
    // type Foo where Foo: Copy = ();
    type_params::where_clause(p);

    p.expect(EQ);
    types::type_(p);
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
