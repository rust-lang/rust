use super::*;

pub(super) const PATTERN_FIRST: TokenSet =
    token_set_union![
        token_set![REF_KW, MUT_KW, L_PAREN, L_BRACK, AMP, UNDERSCORE],
        expressions::LITERAL_FIRST,
        paths::PATH_FIRST,
    ];

pub(super) fn pattern(p: &mut Parser) {
    if let Some(lhs) = atom_pat(p) {
        // test range_pat
        // fn main() {
        //     match 92 { 0 ... 100 => () }
        // }
        if p.at(DOTDOTDOT) {
            let m = lhs.precede(p);
            p.bump();
            atom_pat(p);
            m.complete(p, RANGE_PAT);
        }
    }
}

const PAT_RECOVERY_SET: TokenSet =
    token_set![LET_KW, IF_KW, WHILE_KW, LOOP_KW, MATCH_KW, R_PAREN, COMMA];


fn atom_pat(p: &mut Parser) -> Option<CompletedMarker> {
    let la0 = p.nth(0);
    let la1 = p.nth(1);
    if la0 == REF_KW || la0 == MUT_KW
        || (la0 == IDENT && !(la1 == COLONCOLON || la1 == L_PAREN || la1 == L_CURLY)) {
        return Some(bind_pat(p, true));
    }
    if paths::is_path_start(p) {
        return Some(path_pat(p));
    }

    // test literal_pattern
    // fn main() {
    //     match () {
    //         92 => (),
    //         'c' => (),
    //         "hello" => (),
    //     }
    // }
    match expressions::literal(p) {
        Some(m) => return Some(m),
        None => (),
    }

    let m = match la0 {
        UNDERSCORE => placeholder_pat(p),
        AMP => ref_pat(p),
        L_PAREN => tuple_pat(p),
        L_BRACK => slice_pat(p),
        _ => {
            p.err_recover("expected pattern", PAT_RECOVERY_SET);
            return None;
        }
    };
    Some(m)
}

// test path_part
// fn foo() {
//     let foo::Bar = ();
//     let ::Bar = ();
//     let Bar { .. } = ();
//     let Bar(..) = ();
// }
fn path_pat(p: &mut Parser) -> CompletedMarker {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::expr_path(p);
    let kind = match p.current() {
        L_PAREN => {
            tuple_pat_fields(p);
            TUPLE_STRUCT_PAT
        }
        L_CURLY => {
            field_pat_list(p);
            STRUCT_PAT
        }
        _ => PATH_PAT
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
    assert!(p.at(L_PAREN));
    p.bump();
    while !p.at(EOF) && !p.at(R_PAREN) {
        match p.current() {
            DOTDOT => p.bump(),
            _ => {
                if !p.at_ts(PATTERN_FIRST) {
                    p.error("expected a pattern");
                    break;
                }
                pattern(p)
            }
        }
        if !p.at(R_PAREN) {
            p.expect(COMMA);
        }
    }
    p.expect(R_PAREN);
}

// test field_pat_list
// fn foo() {
//     let S {} = ();
//     let S { f, ref mut g } = ();
//     let S { h: _, ..} = ();
//     let S { h: _, } = ();
// }
fn field_pat_list(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        match p.current() {
            DOTDOT => p.bump(),
            IDENT if p.nth(1) == COLON => {
                p.bump();
                p.bump();
                pattern(p);
            }
            L_CURLY => error_block(p, "expected ident"),
            _ => {
                bind_pat(p, false);
            }
        }
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
    m.complete(p, FIELD_PAT_LIST);
}

// test placeholder_pat
// fn main() { let _ = (); }
fn placeholder_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(UNDERSCORE));
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
    assert!(p.at(AMP));
    let m = p.start();
    p.bump();
    p.eat(MUT_KW);
    pattern(p);
    m.complete(p, REF_PAT)
}

// test tuple_pat
// fn main() {
//     let (a, b, ..) = ();
// }
fn tuple_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(L_PAREN));
    let m = p.start();
    tuple_pat_fields(p);
    m.complete(p, TUPLE_PAT)
}

// test slice_pat
// fn main() {
//     let [a, b, ..] = [];
// }
fn slice_pat(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(L_BRACK));
    let m = p.start();
    p.bump();
    while !p.at(EOF) && !p.at(R_BRACK) {
        match p.current() {
            DOTDOT => p.bump(),
            _ => pattern(p),
        }
        if !p.at(R_BRACK) {
            p.expect(COMMA);
        }
    }
    p.expect(R_BRACK);

    m.complete(p, SLICE_PAT)
}

// test bind_pat
// fn main() {
//     let a = ();
//     let mut b = ();
//     let ref c = ();
//     let ref mut d = ();
//     let e @ _ = ();
//     let ref mut f @ g @ _ = ();
// }
fn bind_pat(p: &mut Parser, with_at: bool) -> CompletedMarker {
    let m = p.start();
    p.eat(REF_KW);
    p.eat(MUT_KW);
    name(p);
    if with_at && p.eat(AT) {
        pattern(p);
    }
    m.complete(p, BIND_PAT)
}
