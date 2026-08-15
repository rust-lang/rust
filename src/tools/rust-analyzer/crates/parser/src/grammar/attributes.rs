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

    if p.expect(T!['[']) {
        meta(p);
        p.expect(T![']']);
    }

    attr.complete(p, ATTR);
}

// test_err meta_recovery
// #![]
// #![p = ]
// #![p::]
// #![p:: =]
// #![unsafe]
// #![unsafe =]

fn cfg_attr_meta(p: &mut Parser<'_>, m: Marker) {
    // test cfg_attr
    // #![cfg_attr(not(foo), unsafe(bar()), cfg_attr(all(true, foo = "bar"), baz = "baz"))]
    p.eat_contextual_kw(T![cfg_attr]);
    p.bump(T!['(']);
    cfg_predicate(p);
    p.expect(T![,]);
    while !p.at(T![')']) && !p.at(EOF) {
        meta(p);
        if !p.eat(T![,]) {
            break;
        }
    }
    p.expect(T![')']);
    m.complete(p, CFG_ATTR_META);
}

const CFG_PREDICATE_FIRST_SET: TokenSet = TokenSet::new(&[T![true], T![false], T![ident]]);

fn cfg_predicate(p: &mut Parser<'_>) {
    let m = p.start();
    if p.eat(T![true]) || p.eat(T![false]) {
        // test cfg_true_false_pred
        // #![cfg(true)]
        // #![cfg(false)]
        m.complete(p, CFG_ATOM);
        return;
    }
    p.expect(T![ident]);
    if p.eat(T![=]) {
        if p.at(T![ident]) {
            // This is required for completion, that inserts an identifier, to work in cases like
            // `#[cfg(key = $0)]`, and also makes sense on itself.

            // test_err key_ident_cfg_predicate
            // #![cfg(key = value)]
            p.err_and_bump("expected a string literal");
        } else {
            // test cfg_key_value_pred
            // #![cfg(key = "value")]
            p.expect(T![string]);
        }
        m.complete(p, CFG_ATOM);
    } else if p.at(T!['(']) {
        // test cfg_composite_pred
        // #![cfg(any(a, all(b = "c", d)))]
        delimited(
            p,
            T!['('],
            T![')'],
            T![,],
            || "expected a cfg predicate".to_owned(),
            CFG_PREDICATE_FIRST_SET,
            |p| {
                if p.at_ts(CFG_PREDICATE_FIRST_SET) {
                    cfg_predicate(p);
                    true
                } else {
                    false
                }
            },
        );
        m.complete(p, CFG_COMPOSITE);
    } else {
        m.complete(p, CFG_ATOM);
    }
}

fn cfg_meta(p: &mut Parser<'_>, m: Marker) {
    // test cfg_meta
    // #![cfg(foo)]
    // #![cfg(foo = "bar",)]
    p.eat_contextual_kw(T![cfg]);
    p.bump(T!['(']);
    cfg_predicate(p);
    p.eat(T![,]);
    p.expect(T![')']);
    m.complete(p, CFG_META);
}

// test metas
// #![simple_ident]
// #![simple::path]
// #![simple_ident_expr = ""]
// #![simple::path::Expr = ""]
// #![simple_ident_tt(a b c)]
// #![simple_ident_tt[a b c]]
// #![simple_ident_tt{a b c}]
// #![simple::path::tt(a b c)]
// #![simple::path::tt[a b c]]
// #![simple::path::tt{a b c}]
// #![unsafe(simple_ident)]
// #![unsafe(simple::path)]
// #![unsafe(simple_ident_expr = "")]
// #![unsafe(simple::path::Expr = "")]
// #![unsafe(simple_ident_tt(a b c))]
// #![unsafe(simple_ident_tt[a b c])]
// #![unsafe(simple_ident_tt{a b c})]
// #![unsafe(simple::path::tt(a b c))]
// #![unsafe(simple::path::tt[a b c])]
// #![unsafe(simple::path::tt{a b c})]
pub(super) fn meta(p: &mut Parser<'_>) {
    let m = p.start();
    if p.eat(T![unsafe]) {
        p.expect(T!['(']);
        meta(p);
        p.expect(T![')']);
        m.complete(p, UNSAFE_META);
        return;
    }

    if p.nth_at(1, T!['(']) {
        if p.at_contextual_kw(T![cfg_attr]) {
            return cfg_attr_meta(p, m);
        } else if p.at_contextual_kw(T![cfg]) {
            return cfg_meta(p, m);
        }
    }

    paths::attr_path(p);

    match p.current() {
        T![=] if !p.at(T![=>]) && !p.at(T![==]) => {
            p.bump(T![=]);
            if expressions::expr(p).is_none() {
                p.error("expected expression");
            }
            m.complete(p, KEY_VALUE_META);
        }
        T!['('] | T!['['] | T!['{'] => {
            items::token_tree(p);
            m.complete(p, TOKEN_TREE_META);
        }
        _ => {
            m.complete(p, PATH_META);
        }
    }
}
