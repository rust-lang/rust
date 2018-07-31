use super::*;

// test expr_literals
// fn foo() {
//     let _ = true;
//     let _ = false;
//     let _ = 1;
//     let _ = 2.0;
//     let _ = b'a';
//     let _ = 'b';
//     let _ = "c";
//     let _ = r"d";
//     let _ = b"e";
//     let _ = br"f";
// }
pub(super) fn literal(p: &mut Parser) -> Option<CompletedMarker> {
    match p.current() {
        TRUE_KW | FALSE_KW | INT_NUMBER | FLOAT_NUMBER | BYTE | CHAR | STRING | RAW_STRING
        | BYTE_STRING | RAW_BYTE_STRING => {
            let m = p.start();
            p.bump();
            Some(m.complete(p, LITERAL))
        }
        _ => None,
    }
}

pub(super) fn expr(p: &mut Parser) {
    let mut lhs = prefix_expr(p);

    while let Some(m) = lhs {
        match p.current() {
            L_PAREN => lhs = Some(call_expr(p, m)),
            _ => break,
        }
    }
}

fn prefix_expr(p: &mut Parser) -> Option<CompletedMarker> {
    match p.current() {
        AMPERSAND => Some(ref_expr(p)),
        _ => atom_expr(p),
    }
}

// test ref_expr
// fn foo() {
//     let _ = &1;
//     let _ = &mut &f();
// }
fn ref_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(AMPERSAND));
    let m = p.start();
    p.bump();
    p.eat(MUT_KW);
    expr(p);
    m.complete(p, REF_EXPR)
}

fn atom_expr(p: &mut Parser) -> Option<CompletedMarker> {
    match literal(p) {
        Some(m) => return Some(m),
        None => (),
    }
    if paths::is_path_start(p) {
        return Some(path_expr(p));
    }

    match p.current() {
        L_PAREN => Some(tuple_expr(p)),
        _ => {
            p.err_and_bump("expected expression");
            None
        }
    }
}

fn tuple_expr(p: &mut Parser) -> CompletedMarker {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.expect(L_PAREN);
    p.expect(R_PAREN);
    m.complete(p, TUPLE_EXPR)
}

// test call_expr
// fn foo() {
//     let _ = f();
//     let _ = f()(1)(1, 2,);
// }
fn call_expr(p: &mut Parser, lhs: CompletedMarker) -> CompletedMarker {
    assert!(p.at(L_PAREN));
    let m = lhs.precede(p);
    arg_list(p);
    m.complete(p, CALL_EXPR)
}

fn arg_list(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.bump();
    while !p.at(R_PAREN) && !p.at(EOF) {
        expr(p);
        if !p.at(R_PAREN) && !p.expect(COMMA) {
            break;
        }
    }
    p.eat(R_PAREN);
    m.complete(p, ARG_LIST);
}

// test path_expr
// fn foo() {
//     let _ = a;
//     let _ = a::b;
//     let _ = ::a::<b>;
// }
fn path_expr(p: &mut Parser) -> CompletedMarker {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::expr_path(p);
    m.complete(p, PATH_EXPR)
}
