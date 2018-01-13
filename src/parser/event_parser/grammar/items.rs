use super::*;

pub(super) fn mod_contents(p: &mut Parser) {
    attributes::inner_attributes(p);
    many(p, |p| {
        skip_to_first(
            p, item_first, mod_contents_item,
            "expected item",
        )
    });
}

fn item_first(p: &Parser) -> bool {
    match p.current() {
        STRUCT_KW | FN_KW | EXTERN_KW | MOD_KW | USE_KW | POUND | PUB_KW => true,
        _ => false,
    }
}

fn mod_contents_item(p: &mut Parser) {
    if item(p) {
        if p.current() == SEMI {
            node(p, ERROR, |p| {
                p.error()
                    .message("expected item, found `;`\n\
                              consider removing this semicolon")
                    .emit();
                p.bump();
            })
        }
    }
}

fn item(p: &mut Parser) -> bool {
    let attrs_start = p.mark();
    attributes::outer_attributes(p);
    visibility(p);
    //    node_if(p, USE_KW, USE_ITEM, use_item)
    // || extern crate_fn
    // || node_if(p, STATIC_KW, STATIC_ITEM, static_item)
    // || node_if(p, CONST_KW, CONST_ITEM, const_item) or const FN!
    // || unsafe trait, impl
    // || node_if(p, FN_KW, FN_ITEM, fn_item)
    // || node_if(p, TYPE_KW, TYPE_ITEM, type_item)
    let item_start = p.mark();
    let item_parsed = node_if(p, [EXTERN_KW, CRATE_KW], EXTERN_CRATE_ITEM, extern_crate_item)
        || node_if(p, MOD_KW, MOD_ITEM, mod_item)
        || node_if(p, USE_KW, USE_ITEM, use_item)
        || node_if(p, STRUCT_KW, STRUCT_ITEM, struct_item)
        || node_if(p, FN_KW, FN_ITEM, fn_item);

    if item_parsed && attrs_start != item_start {
        p.forward_parent(attrs_start, item_start);
    }
    item_parsed
}

fn struct_item(p: &mut Parser) {
    p.expect(IDENT)
        && p.curly_block(|p| comma_list(p, EOF, struct_field));
}

fn extern_crate_item(p: &mut Parser) {
    p.expect(IDENT) && alias(p) && p.expect(SEMI);
}

fn mod_item(p: &mut Parser) {
    if !p.expect(IDENT) {
        return;
    }
    if p.eat(SEMI) {
        return;
    }
    p.curly_block(mod_contents);
}

pub(super) fn is_use_tree_start(kind: SyntaxKind) -> bool {
    kind == STAR || kind == L_CURLY
}

fn use_item(p: &mut Parser) {
    use_tree(p);
    p.expect(SEMI);

    fn use_tree(p: &mut Parser) -> bool{
        if node_if(p, STAR, USE_TREE, |_| ()) {
            return true
        }
        if node_if(p, [COLONCOLON, STAR], USE_TREE, |_| ()) {
           return true
        }
        if [COLONCOLON, L_CURLY].is_ahead(p) || L_CURLY.is_ahead(p) {
            node(p, USE_TREE, |p| {
                p.eat(COLONCOLON);
                p.curly_block(|p| {
                    comma_list(p, EOF, use_tree);
                });
            });
            return true;
        }
        if paths::is_path_start(p) {
            node(p, USE_TREE, |p| {
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
            });
            return true;
        }
        false
    }
}

fn struct_field(p: &mut Parser) -> bool {
    node_if(p, IDENT, STRUCT_FIELD, |p| {
        p.expect(COLON) && p.expect(IDENT);
    })
}

fn fn_item(p: &mut Parser) {
    p.expect(IDENT) && p.expect(L_PAREN) && p.expect(R_PAREN)
        && p.curly_block(|_| ());
}


