use super::*;

// test expr_literals
// fn foo() {
//     let _ = 92;
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
