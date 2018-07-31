use super::*;

pub(super) fn pattern(p: &mut Parser) {
    match p.current() {
        UNDERSCORE => placeholder_pat(p),
        AMPERSAND => ref_pat(p),
        IDENT | REF_KW | MUT_KW => bind_pat(p),
        _ => p.err_and_bump("expected pattern"),
    }
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
    assert!(p.at(AMPERSAND));
    let m = p.start();
    p.bump();
    p.eat(MUT_KW);
    pattern(p);
    m.complete(p, REF_PAT);
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
fn bind_pat(p: &mut Parser) {
    let m = p.start();
    p.eat(REF_KW);
    p.eat(MUT_KW);
    name(p);
    if p.eat(AT) {
        pattern(p);
    }
    m.complete(p, BIND_PAT);
}
