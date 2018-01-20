use super::*;

pub(super) fn mod_contents(p: &mut Parser) {
    attributes::inner_attributes(p);
    while !p.at(EOF) {
        item(p);
    }
}

pub(super) const ITEM_FIRST: TokenSet = token_set![
    EXTERN_KW,
    MOD_KW,
    USE_KW,
    STRUCT_KW,
    FN_KW,
    PUB_KW,
    POUND,
];

fn item(p: &mut Parser) {
    let item = p.start();
    attributes::outer_attributes(p);
    visibility(p);
    let la = p.raw_lookahead(1);
    let item_kind = match p.current() {
        EXTERN_KW if la == CRATE_KW => {
            extern_crate_item(p);
            EXTERN_CRATE_ITEM
        }
        MOD_KW => {
            mod_item(p);
            MOD_ITEM
        }
        USE_KW => {
            use_item(p);
            USE_ITEM
        }
        STRUCT_KW => {
            struct_item(p);
            STRUCT_ITEM
        }
        FN_KW => {
            fn_item(p);
            FN_ITEM
        }
        err_token => {
            item.abandon(p);
            let err = p.start();
            let message = if err_token == SEMI {
                //TODO: if the item is incomplete, this message is misleading
                "expected item, found `;`\n\
                consider removing this semicolon"
            } else {
                "expected item"
            };
            p.error()
                .message(message)
                .emit();
            p.bump();
            err.complete(p, ERROR);
            return;
        }
    };
    item.complete(p, item_kind);
}

fn struct_item(p: &mut Parser) {
    assert!(p.at(STRUCT_KW));
    p.bump();

    if !p.expect(IDENT) {
        return;
    }
    generic_parameters(p);
    match p.current() {
        WHERE_KW => {
            where_clause(p);
            match p.current() {
                SEMI => {
                    p.bump();
                    return;
                }
                L_CURLY => named_fields(p),
                _ => { //TODO: special case `(` error message
                    p.error()
                        .message("expected `;` or `{`")
                        .emit();
                    return;
                }
            }
        }
        SEMI => {
            p.bump();
            return;
        }
        L_CURLY => named_fields(p),
        L_PAREN => {
            pos_fields(p);
            p.expect(SEMI);
        }
        _ => {
            p.error()
                .message("expected `;`, `{`, or `(`")
                .emit();
            return;
        }
    }
}

fn named_fields(p: &mut Parser) {
    p.curly_block(|p| comma_list(p, EOF, |p| {
        named_field(p);
        true
    }));

    fn named_field(p: &mut Parser) {
        let field = p.start();
        visibility(p);
        if p.expect(IDENT) && p.expect(COLON) {
            types::type_ref(p);
        };
        field.complete(p, NAMED_FIELD);
    }
}

fn pos_fields(p: &mut Parser) {
    if !p.expect(L_PAREN) {
        return;
    }
    comma_list(p, R_PAREN, |p| {
        pos_field(p);
        true
    });
    p.expect(R_PAREN);

    fn pos_field(p: &mut Parser) {
        let pos_field = p.start();
        visibility(p);
        types::type_ref(p);
        pos_field.complete(p, POS_FIELD);
    }
}

fn generic_parameters(_: &mut Parser) {}

fn where_clause(_: &mut Parser) {}

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
        p.curly_block(mod_contents);
    }
}

pub(super) fn is_use_tree_start(kind: SyntaxKind) -> bool {
    kind == STAR || kind == L_CURLY
}

fn use_item(p: &mut Parser) {
    assert!(p.at(USE_KW));
    p.bump();

    use_tree(p);
    p.expect(SEMI);

    fn use_tree(p: &mut Parser) -> bool {
        let la = p.raw_lookahead(1);
        let m = p.start();
        match (p.current(), la) {
            (STAR, _) => {
                p.bump();
            }
            (COLONCOLON, STAR) => {
                p.bump();
                p.bump();
            }
            (L_CURLY, _) | (COLONCOLON, L_CURLY) => {
                if p.at(COLONCOLON) {
                    p.bump();
                }
                p.curly_block(|p| {
                    comma_list(p, EOF, use_tree);
                });
            }
            _ if paths::is_path_start(p) => {
                paths::use_path(p);
                match p.current() {
                    AS_KW => {
                        alias(p);
                    }
                    COLONCOLON => {
                        p.bump();
                        match p.current() {
                            STAR => {
                                p.bump();
                            }
                            L_CURLY => {
                                p.curly_block(|p| {
                                    comma_list(p, EOF, use_tree);
                                });
                            }
                            _ => {
                                // is this unreachable?
                                p.error()
                                    .message("expected `{` or `*`")
                                    .emit();
                            }
                        }
                    }
                    _ => (),
                }
            }
            _ => {
                m.abandon(p);
                return false
            },
        }
        m.complete(p, USE_TREE);
        return true;
    }
}


fn fn_item(p: &mut Parser) {
    assert!(p.at(FN_KW));
    p.bump();

    p.expect(IDENT) && p.expect(L_PAREN) && p.expect(R_PAREN)
        && p.curly_block(|_| ());
}


