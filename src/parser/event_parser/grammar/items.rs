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
        STRUCT_KW | FN_KW | EXTERN_KW | MOD_KW | USE_KW | POUND => true,
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
    attributes::outer_attributes(p);
    visibility(p);
    //    node_if(p, USE_KW, USE_ITEM, use_item)
    // || extern crate_fn
    // || node_if(p, STATIC_KW, STATIC_ITEM, static_item)
    // || node_if(p, CONST_KW, CONST_ITEM, const_item) or const FN!
    // || unsafe trait, impl
    // || node_if(p, FN_KW, FN_ITEM, fn_item)
    // || node_if(p, TYPE_KW, TYPE_ITEM, type_item)
    node_if(p, [EXTERN_KW, CRATE_KW], EXTERN_CRATE_ITEM, extern_crate_item)
        || node_if(p, MOD_KW, MOD_ITEM, mod_item)
        || node_if(p, USE_KW, USE_ITEM, use_item)
        || node_if(p, STRUCT_KW, STRUCT_ITEM, struct_item)
        || node_if(p, FN_KW, FN_ITEM, fn_item)
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

fn use_item(p: &mut Parser) {
    paths::use_path(p);
    p.expect(SEMI);
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


