use super::*;

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
    if !literal(p) {
        p.error("expected expression");
    }
}
