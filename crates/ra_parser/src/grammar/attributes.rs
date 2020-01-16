//! FIXME: write short doc here

use super::*;

pub(super) fn inner_attributes(p: &mut Parser) {
    while p.at(T![#]) && p.nth(1) == T![!] {
        attribute(p, true)
    }
}

pub(super) fn with_outer_attributes(
    p: &mut Parser,
    f: impl Fn(&mut Parser) -> Option<CompletedMarker>,
) -> bool {
    let am = p.start();
    let has_attrs = p.at(T![#]);
    attributes::outer_attributes(p);
    let cm = f(p);
    let success = cm.is_some();

    match (has_attrs, cm) {
        (true, Some(cm)) => {
            let kind = cm.kind();
            cm.undo_completion(p).abandon(p);
            am.complete(p, kind);
        }
        _ => am.abandon(p),
    }

    success
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

        match p.current() {
            T![=] => {
                p.bump(T![=]);
                if expressions::literal(p).is_none() {
                    p.error("expected literal");
                }
            }
            T!['('] | T!['['] | T!['{'] => items::token_tree(p),
            _ => {}
        }

        if !p.eat(T![']']) {
            p.error("expected `]`");
        }
    } else {
        p.error("expected `[`");
    }
    attr.complete(p, ATTR);
}
