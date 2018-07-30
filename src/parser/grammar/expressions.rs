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
pub(super) fn literal(p: &mut Parser) -> bool {
    match p.current() {
        TRUE_KW | FALSE_KW | INT_NUMBER | FLOAT_NUMBER | BYTE | CHAR | STRING | RAW_STRING
        | BYTE_STRING | RAW_BYTE_STRING => {
            let lit = p.start();
            p.bump();
            lit.complete(p, LITERAL);
            true
        }
        _ => false,
    }
}

pub(super) fn expr(p: &mut Parser) {
    if literal(p) {
        return;
    }
    if paths::is_path_start(p) {
        return path_expr(p);
    }

    match p.current() {
        L_PAREN => tuple_expr(p),
        _ => p.error("expected expression"),
    }
}

fn tuple_expr(p: &mut Parser) {
    assert!(p.at(L_PAREN));
    let m = p.start();
    p.expect(L_PAREN);
    p.expect(R_PAREN);
    m.complete(p, TUPLE_EXPR);
}

// test path_expr
// fn foo() {
//     let _ = a;
//     let _ = a::b;
//     let _ = ::a::<b>;
// }
fn path_expr(p: &mut Parser) {
    assert!(paths::is_path_start(p));
    let m = p.start();
    paths::expr_path(p);
    m.complete(p, PATH_EXPR);
}
