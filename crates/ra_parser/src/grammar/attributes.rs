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
    p.bump(T![#]);

    if inner {
        assert!(p.at(T![!]));
        p.bump(T![!]);
    }

    if p.eat(T!['[']) {
        paths::use_path(p);

        let is_delimiter = |p: &mut Parser| match p.current() {
            T!['('] | T!['['] | T!['{'] => true,
            _ => false,
        };

        if p.eat(T![=]) {
            if expressions::literal(p).is_none() {
                p.error("expected literal");
            }
        } else if is_delimiter(p) {
            items::token_tree(p);
        }

        if !p.eat(T![']']) {
            p.error("expected `]`");
        }
    } else {
        p.error("expected `[`");
    }
    attr.complete(p, ATTR);
}
