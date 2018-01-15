use super::*;

enum AttrKind {
    Inner, Outer
}

pub(super) fn inner_attributes(p: &mut Parser) {
    repeat(p, |p| attribute(p, AttrKind::Inner))
}

pub(super) fn outer_attributes(p: &mut Parser) {
    repeat(p, |p| attribute(p, AttrKind::Outer))
}


fn attribute(p: &mut Parser, kind: AttrKind) -> bool {
    fn attr_tail(p: &mut Parser) {
        meta_item(p) && p.expect(R_BRACK);
    }

    match kind {
        AttrKind::Inner => node_if(p, [POUND, EXCL, L_BRACK], ATTR, attr_tail),
        AttrKind::Outer => node_if(p, [POUND, L_BRACK], ATTR, attr_tail),
    }
}

fn meta_item(p: &mut Parser) -> bool {
    node_if(p, IDENT, META_ITEM, |p| {
        if p.eat(EQ) {
            if !expressions::literal(p) {
                p.error()
                    .message("expected literal")
                    .emit();
            }
        } else if p.eat(L_PAREN) {
            comma_list(p, R_PAREN, meta_item_inner);
            p.expect(R_PAREN);
        }
    })
}

fn meta_item_inner(p: &mut Parser) -> bool {
    meta_item(p) || expressions::literal(p)
}

