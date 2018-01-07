use super::*;

pub(super) fn inner_attributes(p: &mut Parser) {
    many(p, |p| attribute(p, true))
}

pub(super) fn outer_attributes(_: &mut Parser) {
}


fn attribute(p: &mut Parser, inner: bool) -> bool {
    let attr_start = inner && p.lookahead(&[POUND, EXCL, L_BRACK])
        || !inner && p.lookahead(&[POUND, L_BRACK]);
    if !attr_start {
        return false;
    }
    node(p, ATTR, |p| {
        p.bump_n(if inner { 3 } else { 2 });
        meta_item(p) && p.expect(R_BRACK);
    });
    true
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

