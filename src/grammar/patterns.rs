use super::*;

pub(super) fn pattern(p: &mut Parser) {
    let la0 = p.nth(0);
    let la1 = p.nth(1);
    if la0 == REF_KW || la0 == MUT_KW
        || (la0 == IDENT && !(la1 == COLONCOLON || la1 == L_PAREN || la1 == L_CURLY)) {
        bind_pat(p, true);
        return;
    }
    if paths::is_path_start(p) {
        path_pat(p);
        return;
    }

    match la0 {
        UNDERSCORE => placeholder_pat(p),
        AMP => ref_pat(p),
        L_PAREN => tuple_pat(p),
        _ => p.err_and_bump("expected pattern"),
    }
}

// test path_part
// fn foo() {
//     let foo::Bar = ();
//     let ::Bar = ();
//     let Bar { .. } = ();
//     let Bar(..) = ();
// }
fn path_pat(p: &mut Parser) {
    let m = p.start();
    paths::expr_path(p);
    let kind = match p.current() {
        L_PAREN => {
            tuple_pat_fields(p);
            TUPLE_STRUCT_PAT
        }
        L_CURLY => {
            struct_pat_fields(p);
            STRUCT_PAT
        }
        _ => PATH_PAT
    };
    m.complete(p, kind);
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
            _ => pattern(p),
        }
        if !p.at(R_PAREN) {
            p.expect(COMMA);
        }
    }
    p.expect(R_PAREN);
}

// test struct_pat_fields
// fn foo() {
//     let S {} = ();
//     let S { f, ref mut g } = ();
//     let S { h: _, ..} = ();
//     let S { h: _, } = ();
// }
fn struct_pat_fields(p: &mut Parser) {
    assert!(p.at(L_CURLY));
    p.bump();
    while !p.at(EOF) && !p.at(R_CURLY) {
        match p.current() {
            DOTDOT => p.bump(),
            IDENT if p.nth(1) == COLON => {
                p.bump();
                p.bump();
                pattern(p);
            }
            _ => bind_pat(p, false),
        }
        if !p.at(R_CURLY) {
            p.expect(COMMA);
        }
    }
    p.expect(R_CURLY);
}

// test placeholder_pat
// fn main() { let _ = (); }
fn placeholder_pat(p: &mut Parser) {
    assert!(p.at(UNDERSCORE));
    let m = p.start();
    p.bump();
    m.complete(p, PLACEHOLDER_PAT);
}

// test ref_pat
// fn main() {
//     let &a = ();
//     let &mut b = ();
// }
fn ref_pat(p: &mut Parser) {
    assert!(p.at(AMP));
    let m = p.start();
    p.bump();
    p.eat(MUT_KW);
    pattern(p);
    m.complete(p, REF_PAT);
}

// test tuple_pat
// fn main() {
//     let (a, b, ..) = ();
// }
fn tuple_pat(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    let m = p.start();
    tuple_pat_fields(p);
    m.complete(p, TUPLE_PAT);
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
fn bind_pat(p: &mut Parser, with_at: bool) {
    let m = p.start();
    p.eat(REF_KW);
    p.eat(MUT_KW);
    name(p);
    if with_at && p.eat(AT) {
        pattern(p);
    }
    m.complete(p, BIND_PAT);
}
