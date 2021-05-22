use super::*;

pub(super) fn inner_attrs(p: &mut Parser) {
    while p.at(T![#]) && p.nth(1) == T![!] {
        attr(p, true)
    }
}

pub(super) fn outer_attrs(p: &mut Parser) {
    while p.at(T![#]) {
        attr(p, false)
    }
}

pub(super) fn meta(p: &mut Parser) {
    paths::use_path(p);

    match p.current() {
        T![=] => {
            p.bump(T![=]);
            if expressions::expr(p).0.is_none() {
                p.error("expected expression");
            }
        }
        T!['('] | T!['['] | T!['{'] => items::token_tree(p),
        _ => {}
    }
}

fn attr(p: &mut Parser, inner: bool) {
    let attr = p.start();
    assert!(p.at(T![#]));
    p.bump(T![#]);

    if inner {
        assert!(p.at(T![!]));
        p.bump(T![!]);
    }

    if p.eat(T!['[']) {
        meta(p);

        if !p.eat(T![']']) {
            p.error("expected `]`");
        }
    } else {
        p.error("expected `[`");
    }
    attr.complete(p, ATTR);
}
