use super::*;

pub(super) const PATTERN_FIRST: TokenSet = expressions::LITERAL_FIRST
    .union(paths::PATH_FIRST)
    .union(token_set![REF_KW, MUT_KW, L_PAREN, L_BRACK, AMP, UNDERSCORE, MINUS]);

pub(super) fn pattern(p: &mut Parser) {
    pattern_r(p, PAT_RECOVERY_SET);
}

/// Parses a pattern list separated by pipes `|`
pub(super) fn pattern_list(p: &mut Parser) {
    pattern_list_r(p, PAT_RECOVERY_SET)
}

/// Parses a pattern list separated by pipes `|`
/// using the given `recovery_set`
pub(super) fn pattern_list_r(p: &mut Parser, recovery_set: TokenSet) {
    p.eat(T![|]);
    pattern_r(p, recovery_set);

    while p.eat(T![|]) {
        pattern_r(p, recovery_set);
    }
}

pub(super) fn pattern_r(p: &mut Parser, recovery_set: TokenSet) {
    if let Some(lhs) = atom_pat(p, recovery_set) {
        // test range_pat
        // fn main() {
        //     match 92 {
        //         0 ... 100 => (),
        //         101 ..= 200 => (),
        //         200 .. 301=> (),
        //     }
        // }
        if p.at(T![...]) || p.at(T![..=]) || p.at(T![..]) {
            let m = lhs.precede(p);
            p.bump();
            atom_pat(p, recovery_set);
            m.complete(p, RANGE_PAT);
        }
        // test marco_pat
        // fn main() {
        //     let m!(x) = 0;
        // }
        else if lhs.kind() == PATH_PAT && p.at(T![!]) {
            let m = lhs.precede(p);
            items::macro_call_after_excl(p);
            m.complete(p, MACRO_CALL);
        }
    }
}

const PAT_RECOVERY_SET: TokenSet =
    token_set![LET_KW, IF_KW, WHILE_KW, LOOP_KW, MATCH_KW, R_PAREN, COMMA];

fn atom_pat(p: &mut Parser, recovery_set: TokenSet) -> Option<CompletedMarker> {
    let la0 = p.nth(0);
    let la1 = p.nth(1);
    if la0 == T![ref]
        || la0 == T![mut]
        || la0 == T![box]
        || (la0 == IDENT && !(la1 == T![::] || la1 == T!['('] || la1 == T!['{'] || la1 == T![!]))
    {
        return Some(bind_pat(p, true));
    }
    if paths::is_use_path_start(p) {
        return Some(path_pat(p));
    }

    if is_literal_pat_start(p) {
        return Some(literal_pat(p));
    }

    let m = match la0 {
        T![_] => placeholder_pat(p),
        T![&] => ref_pat(p),
        T!['('] => tuple_pat(p),
        T!['['] => slice_pat(p),
        _ => {
            p.err_recover("expected pattern", recovery_set);
            return None;
        }
    };
    Some(m)
}

fn is_literal_pat_start(p: &mut Parser) -> bool {
    p.at(T![-]) && (p.nth(1) == INT_NUMBER || p.nth(1) == FLOAT_NUMBER)
        || p.at_ts(expressions::LITERAL_FIRST)
}

// test literal_pattern
// fn main() {
//     match () {
//         -1 => (),
//         92 => (),
//         'c' => (),
//         "hello" => (),
//     }
// }
fn literal_pat(p: &mut Parser) -> CompletedMarker {
    assert!(is_literal_pat_start(p));
    let m = p.start();
    if p.at(T![-]) {
        p.bump();
    }
    expressions::literal(p);
    m.complete(p, LITERAL_PAT)
}

// test path_part
// fn foo() {
//     let foo::Bar = ();
//     let ::Bar = ();
//     let Bar { .. } = ();
//     let Bar(..) = ();
// }
fn path_pat(p: &mut Parser) -> CompletedMarker {
    assert!(paths::is_use_path_start(p));
    let m = p.start();
    paths::expr_path(p);
    let kind = match p.current() {
        T!['('] => {
            tuple_pat_fields(p);
            TUPLE_STRUCT_PAT
        }
        T!['{'] => {
            field_pat_list(p);
            STRUCT_PAT
        }
        _ => PATH_PAT,
    };
    m.complete(p, kind)
}

// test tuple_pat_fields
// fn foo() {
//     let S() = ();
//     let S(_) = ();
//     let S(_,) = ();
//     let S(_, .. , x) = ();
// }
fn tuple_pat_fields(p: &mut Parser) {
    assert!(p.at(T!['(']));
    p.bump();
    pat_list(p, T![')']);
    p.expect(T![')']);
}

// test field_pat_list
// fn foo() {
//     let S {} = ();
//     let S { f, ref mut g } = ();
//     let S { h: _, ..} = ();
//     let S { h: _, } = ();
// }
fn field_pat_list(p: &mut Parser) {
    assert!(p.at(T!['{']));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(T!['}']) {
        match p.current() {
            T![..] => p.bump(),
            IDENT if p.nth(1) == T![:] => field_pat(p),
            T!['{'] => error_block(p, "expected ident"),
            _ => {
                bind_pat(p, false);
            }
        }
        if !p.at(T!['}']) {
            p.expect(T![,]);
        }
    }
    p.expect(T!['}']);
    m.complete(p, FIELD_PAT_LIST);
}

fn field_pat(p: &mut Parser) {
    assert!(p.at(IDENT));
    assert!(p.nth(1) == T![:]);

    let m = p.start();
    name(p);
    p.bump();
    pattern(p);
    m.complete(p, FIELD_PAT);
}

// test placeholder_pat
// fn main() { let _ = (); }
fn placeholder_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![_]));
    let m = p.start();
    p.bump();
    m.complete(p, PLACEHOLDER_PAT)
}

// test ref_pat
// fn main() {
//     let &a = ();
//     let &mut b = ();
// }
fn ref_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T![&]));
    let m = p.start();
    p.bump();
    p.eat(T![mut]);
    pattern(p);
    m.complete(p, REF_PAT)
}

// test tuple_pat
// fn main() {
//     let (a, b, ..) = ();
// }
fn tuple_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T!['(']));
    let m = p.start();
    tuple_pat_fields(p);
    m.complete(p, TUPLE_PAT)
}

// test slice_pat
// fn main() {
//     let [a, b, ..] = [];
// }
fn slice_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(T!['[']));
    let m = p.start();
    p.bump();
    pat_list(p, T![']']);
    p.expect(T![']']);
    m.complete(p, SLICE_PAT)
}

fn pat_list(p: &mut Parser, ket: SyntaxKind) {
    while !p.at(EOF) && !p.at(ket) {
        match p.current() {
            T![..] => p.bump(),
            _ => {
                if !p.at_ts(PATTERN_FIRST) {
                    p.error("expected a pattern");
                    break;
                }
                pattern(p)
            }
        }
        if !p.at(ket) {
            p.expect(T![,]);
        }
    }
}

// test bind_pat
// fn main() {
//     let a = ();
//     let mut b = ();
//     let ref c = ();
//     let ref mut d = ();
//     let e @ _ = ();
//     let ref mut f @ g @ _ = ();
//     let box i = Box::new(1i32);
// }
fn bind_pat(p: &mut Parser, with_at: bool) -> CompletedMarker {
    let m = p.start();
    p.eat(T![box]);
    p.eat(T![ref]);
    p.eat(T![mut]);
    name(p);
    if with_at && p.eat(T![@]) {
        pattern(p);
    }
    m.complete(p, BIND_PAT)
}
