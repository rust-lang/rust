use super::*;

pub(super) fn mod_contents(p: &mut Parser) {
    attributes::inner_attributes(p);
    while !p.at(EOF) {
        item(p);
    }
}

fn item(p: &mut Parser) {
    let attrs_start = p.mark();
    attributes::outer_attributes(p);
    visibility(p);
    let la = p.raw_lookahead(1);
    let item_start = p.mark();
    match p.current() {
        EXTERN_KW if la == CRATE_KW => extern_crate_item(p),
        MOD_KW => mod_item(p),
        USE_KW => use_item(p),
        STRUCT_KW => struct_item(p),
        FN_KW => fn_item(p),
        err_token => {
            p.start(ERROR);
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
            p.finish();
            return;
        }
    };
    p.forward_parent(attrs_start, item_start);
}

fn struct_item(p: &mut Parser) {
    p.start(STRUCT_ITEM);

    assert!(p.at(STRUCT_KW));
    p.bump();

    struct_inner(p);
    p.finish();

    fn struct_inner(p: &mut Parser) {
        if !p.expect(IDENT) {
            p.finish();
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
                tuple_fields(p);
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
}

fn named_fields(p: &mut Parser) {
    p.curly_block(|p| comma_list(p, EOF, |p| {
        named_field(p);
        true
    }));

    fn named_field(p: &mut Parser) {
        p.start(NAMED_FIELD);
        visibility(p);
        if p.expect(IDENT) && p.expect(COLON) {
            types::type_ref(p);
        };
        p.finish()
    }
}

fn tuple_fields(p: &mut Parser) {
    if !p.expect(L_PAREN) {
        return;
    }
    comma_list(p, R_PAREN, |p| {
        tuple_field(p);
        true
    });
    p.expect(R_PAREN);

    fn tuple_field(p: &mut Parser) {
        p.start(POS_FIELD);
        visibility(p);
        types::type_ref(p);
        p.finish();
    }
}

fn generic_parameters(_: &mut Parser) {}

fn where_clause(_: &mut Parser) {}

fn extern_crate_item(p: &mut Parser) {
    p.start(EXTERN_CRATE_ITEM);

    assert!(p.at(EXTERN_KW));
    p.bump();

    assert!(p.at(CRATE_KW));
    p.bump();

    p.expect(IDENT) && alias(p) && p.expect(SEMI);
    p.finish();
}

fn mod_item(p: &mut Parser) {
    p.start(MOD_ITEM);

    assert!(p.at(MOD_KW));
    p.bump();

    if p.expect(IDENT) && !p.eat(SEMI) {
        p.curly_block(mod_contents);
    }
    p.finish()
}

pub(super) fn is_use_tree_start(kind: SyntaxKind) -> bool {
    kind == STAR || kind == L_CURLY
}

fn use_item(p: &mut Parser) {
    p.start(USE_ITEM);

    assert!(p.at(USE_KW));
    p.bump();
    use_tree(p);
    p.expect(SEMI);
    p.finish();

    fn use_tree(p: &mut Parser) -> bool {
        let la = p.raw_lookahead(1);
        match (p.current(), la) {
            (STAR, _) => {
                p.start(USE_TREE);
                p.bump();
            }
            (COLONCOLON, STAR) => {
                p.start(USE_TREE);
                p.bump();
                p.bump();
            }
            (L_CURLY, _) | (COLONCOLON, L_CURLY) => {
                p.start(USE_TREE);
                if p.at(COLONCOLON) {
                    p.bump();
                }
                p.curly_block(|p| {
                    comma_list(p, EOF, use_tree);
                });
            }
            _ if paths::is_path_start(p) => {
                p.start(USE_TREE);
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
            _ => return false,
        }
        p.finish();
        return true;
    }
}


fn fn_item(p: &mut Parser) {
    p.start(FN_ITEM);

    assert!(p.at(FN_KW));
    p.bump();

    p.expect(IDENT) && p.expect(L_PAREN) && p.expect(R_PAREN)
        && p.curly_block(|_| ());
    p.finish();
}


