use super::*;

pub(super) const ATTRIBUTE_FIRST: TokenSet = TokenSet::new(&[T![#]]);

pub(super) fn inner_attrs(p: &mut Parser<'_>) {
    while p.at(T![#]) && p.nth(1) == T![!] {
        attr(p, true);
    }
}

pub(super) fn outer_attrs(p: &mut Parser<'_>) {
    while p.at(T![#]) {
        attr(p, false);
    }
}

fn attr(p: &mut Parser<'_>, inner: bool) {
    assert!(p.at(T![#]));

    let attr = p.start();
    p.bump(T![#]);

    if inner {
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

pub(super) fn meta(p: &mut Parser<'_>) {
    let meta = p.start();
    paths::use_path(p);

    match p.current() {
        T![=] => {
            p.bump(T![=]);
            if !expressions::expr(p) {
                p.error("expected expression");
            }
        }
        T!['('] | T!['['] | T!['{'] => items::token_tree(p),
        _ => {}
    }

    meta.complete(p, META);
}
