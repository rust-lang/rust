use super::*;

#[derive(PartialEq, Eq)]
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
    if p.at(POUND) {
        if kind == AttrKind::Inner && p.raw_lookahead(1) != EXCL {
            return false;
        }
        let attr = p.start();
        p.bump();
        if kind == AttrKind::Inner {
            p.bump();
        }
        p.expect(L_BRACK) && meta_item(p) && p.expect(R_BRACK);
        attr.complete(p, ATTR);
        true
    } else {
        false
    }
}

fn meta_item(p: &mut Parser) -> bool {
    if p.at(IDENT) {
        let meta_item = p.start();
        p.bump();
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
        meta_item.complete(p, META_ITEM);
        true
    } else {
        false
    }

}

fn meta_item_inner(p: &mut Parser) -> bool {
    meta_item(p) || expressions::literal(p)
}

