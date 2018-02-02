use super::*;

mod structs;
mod use_item;

pub(super) fn mod_contents(p: &mut Parser, stop_on_r_curly: bool) {
    attributes::inner_attributes(p);
    while !p.at(EOF) && !(stop_on_r_curly && p.at(R_CURLY)) {
        item(p);
    }
}

pub(super) const ITEM_FIRST: TokenSet = token_set![
    EXTERN_KW, MOD_KW, USE_KW, STRUCT_KW, ENUM_KW, FN_KW, PUB_KW, POUND
];

fn item(p: &mut Parser) {
    let item = p.start();
    attributes::outer_attributes(p);
    visibility(p);
    let la = p.nth(1);
    let item_kind = match p.current() {
        USE_KW => {
            use_item::use_item(p);
            USE_ITEM
        }
        EXTERN_KW if la == CRATE_KW => {
            extern_crate_item(p);
            EXTERN_CRATE_ITEM
        }
        EXTERN_KW => {
            abi(p);
            match p.current() {
                FN_KW => {
                    fn_item(p);
                    FN_ITEM
                }
                L_CURLY => {
                    extern_block(p);
                    EXTERN_BLOCK
                }
                _ => {
                    item.abandon(p);
                    p.error().message("expected `fn` or `{`").emit();
                    return;
                }
            }
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
        FN_KW => {
            fn_item(p);
            FN_ITEM
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

    p.expect(IDENT) && alias(p) && p.expect(SEMI);
}

fn mod_item(p: &mut Parser) {
    assert!(p.at(MOD_KW));
    p.bump();

    if p.expect(IDENT) && !p.eat(SEMI) {
        if p.expect(L_CURLY) {
            mod_contents(p, true);
            p.expect(R_CURLY);
        }
    }
}

fn extern_block(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    p.expect(R_CURLY);
}

fn abi(p: &mut Parser) {
    assert!(p.at(EXTERN_KW));
    let abi = p.start();
    p.bump();
    match p.current() {
        STRING | RAW_STRING => p.bump(),
        _ => (),
    }
    abi.complete(p, ABI);
}

fn fn_item(p: &mut Parser) {
    assert!(p.at(FN_KW));
    p.bump();

    p.expect(IDENT);
    if p.at(L_PAREN) {
        fn_value_parameters(p);
    } else {
        p.error().message("expected function arguments").emit();
    }

    if p.at(L_CURLY) {
        p.expect(L_CURLY);
        p.expect(R_CURLY);
    }

    fn fn_value_parameters(p: &mut Parser) {
        assert!(p.at(L_PAREN));
        p.bump();
        p.expect(R_PAREN);
    }
}
