use super::*;

pub(super) fn literal(p: &mut Parser) -> bool {
    match p.current() {
        TRUE_KW | FALSE_KW |
        INT_NUMBER | FLOAT_NUMBER |
        BYTE | CHAR |
        STRING | RAW_STRING | BYTE_STRING | RAW_BYTE_STRING => {
            p.start(LITERAL);
            p.bump();
            p.finish();
            true
        }
        _ => false
    }
}
