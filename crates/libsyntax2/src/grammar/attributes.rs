use super::*;

pub(super) fn inner_attributes(p: &mut Parser) {
    while p.current() == POUND && p.nth(1) == EXCL {
        attribute(p, true)
    }
}

pub(super) fn outer_attributes(p: &mut Parser) {
    while p.at(POUND) {
        attribute(p, false)
    }
}

fn attribute(p: &mut Parser, inner: bool) {
    let attr = p.start();
    assert!(p.at(POUND));
    p.bump();

    if inner {
        assert!(p.at(EXCL));
        p.bump();
    }

    if p.expect(L_BRACK) {
        meta_item(p);
        p.expect(R_BRACK);
    }
    attr.complete(p, ATTR);
}

fn meta_item(p: &mut Parser) {
    if p.at(IDENT) {
        let meta_item = p.start();
        p.bump();
        match p.current() {
            EQ => {
                p.bump();
                if expressions::literal(p).is_none() {
                    p.error("expected literal");
                }
            }
            L_PAREN => meta_item_arg_list(p),
            _ => (),
        }
        meta_item.complete(p, META_ITEM);
    } else {
        p.error("expected attribute value");
    }
}

fn meta_item_arg_list(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    p.bump();
    loop {
        match p.current() {
            EOF | R_PAREN => break,
            IDENT => meta_item(p),
            c => if expressions::literal(p).is_none() {
                let message = "expected attribute";

                if items::ITEM_FIRST.contains(c) {
                    p.error(message);
                    return;
                }

                let err = p.start();
                p.error(message);
                p.bump();
                err.complete(p, ERROR);
                continue;
            },
        }
        if !p.at(R_PAREN) {
            p.expect(COMMA);
        }
    }
    p.expect(R_PAREN);
}
