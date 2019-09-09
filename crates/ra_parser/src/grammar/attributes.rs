use super::*;

pub(super) fn inner_attributes(p: &mut Parser) {
    while p.at(T![#]) && p.nth(1) == T![!] {
        attribute(p, true)
    }
}

pub(super) fn outer_attributes(p: &mut Parser) {
    while p.at(T![#]) {
        attribute(p, false)
    }
}

fn attribute(p: &mut Parser, inner: bool) {
    let attr = p.start();
    assert!(p.at(T![#]));
    p.bump();

    if inner {
        assert!(p.at(T![!]));
        p.bump();
    }

    if p.at(T!['[']) {
        items::token_tree(p);
    } else {
        p.error("expected `[`");
    }
    attr.complete(p, ATTR);
}
